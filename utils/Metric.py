import time
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import pickle as pk
import functools
import wandb


def save_model(model: nn.Module, save_path, run):
    torch.save(model.state_dict(), f'{save_path}_{run}.pth')


def load_model(model: nn.Module, load_path, run):
    model_dict = torch.load(f"{load_path}_{run}.pth")
    model.load_state_dict(model_dict)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10, save_path=None, logger=None,
                 model: nn.Module = None,
                 run=0):
        self.max_round = max_round
        self.num_round = 0
        self.run = run

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.save_path = save_path
        self.logger = logger
        self.model = model

    def early_stop_check(self, curr_val):
        state = None
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            save_model(self.model, self.save_path, self.run)
            state = 'better'
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            save_model(self.model, self.save_path, self.run)
            state = 'better'
        else:
            self.num_round += 1
        self.epoch_count += 1
        if self.num_round <= self.max_round:
            state = 'continue' if state is None else state
        else:
            state = 'break'
        return state


class MetricManager:
    def __init__(self, path, logger, wandb_config, k_list=None):
        if k_list is None:
            k_list = [10, 50, 100]
        self.path = path
        self.logger = logger
        self.k_list = k_list
        metric_names = Metrics(k_list).get_metric_names()
        self.avg_metric = dict(zip(metric_names, [0.0] * len(metric_names)))
        self.wandb_config = wandb_config
        self.run = 0

    def init_run(self, early_stopper):
        wandb_args = deepcopy(self.wandb_config)
        wandb_args['tags'] = wandb_args.get('tags', []) + ['run']
        wandb_args['name'] = wandb_args['name'] + f'_run{self.run}'
        # wandb_args['id'] = wandb_args['name']
        self.early_stopper = early_stopper
        self.run_wandb = wandb.init(**wandb_args)
        self.run_logger = MetricRunManager(f'{self.path}_{self.run}.pkl', self.logger, self.k_list, self.run_wandb,
                                           self.early_stopper)

    def watch(self, model: nn.Module):
        self.run_wandb.watch(model, log='all', log_freq=100)

    def finish_run(self):
        """
        1) get the best test scores. Put it into wandb_run and average_metric
        2) save the data of the run
        3) release corresponding resources and update the run count
        """
        run_best_scores = self.run_logger.get_final_scores()
        for metric in self.avg_metric:
            self.avg_metric[metric] += run_best_scores['test'][metric]
            self.run_wandb.summary[metric] = run_best_scores['test'][metric]
        self.run_logger.save()
        self.run_logger, self.early_stopper = None, None
        self.run_wandb.finish()
        self.run += 1
        return deepcopy(run_best_scores)

    def finish(self):
        wandb_args = deepcopy(self.wandb_config)
        wandb_args['tags'] = wandb_args.get('tags', []) + ['average']
        # wandb_args['id'] = wandb_args['name']
        final_run = wandb.init(**wandb_args)
        for metric in self.avg_metric:
            self.avg_metric[metric] /= self.run
            self.avg_metric[metric] = float(f"{self.avg_metric[metric]:.4f}")
        self.logger.info(f"{self.avg_metric}")
        final_run.summary.update(self.avg_metric)
        return deepcopy(self.avg_metric)

    def update(self, target, pred, label, dtype):
        self.run_logger.update(target, pred, label, dtype)

    def calculate_metric(self, dtype, loss=0):
        return self.run_logger.calculate_metric(dtype, loss)

    def init_epoch(self):
        self.run_logger.init_epoch()

    def finish_epoch(self):
        return self.run_logger.finish_epoch()

    def info_epoch(self):
        self.run_logger.info_epoch()


class MetricRunManager:
    def __init__(self, path, logger, k_list=[10, 50, 100], wandb_run: wandb.run = None,
                 early_stopper: EarlyStopMonitor = None):
        self.path = path
        self.logger = logger
        self.metrics = Metrics(k_list)
        self.metric_names = ['loss'] + self.metrics.get_metric_names()
        self.score_template = dict(zip(self.metric_names, [0] * len(self.metric_names)))
        self.final_scores = {'train': deepcopy(self.score_template), 'val': deepcopy(self.score_template),
                             'test': deepcopy(self.score_template)}
        self.wandb_run = wandb_run
        self.epoch = 0
        self.early_stopper = early_stopper

    def init_epoch(self):
        self.temp_score = {'train': deepcopy(self.score_template), 'val': deepcopy(self.score_template),
                           'test': deepcopy(self.score_template)}
        self.start_time = time.time()
        self.data_cnt = {'train': 0, 'val': 0, 'test': 0}

    def update(self, target, pred, label, dtype):
        self.data_cnt[dtype] += len(target)
        scores = self.metrics.get_metrics(pred, label,
                                          mean_reduce=False)
        for metric in scores:
            self.temp_score[dtype][metric] += scores[metric]

    def calculate_metric(self, dtype, loss=0):
        """
        calculate the result of an epoch's metric
        """
        for metric in self.temp_score[dtype]:
            self.temp_score[dtype][metric] /= self.data_cnt[dtype]
        self.temp_score[dtype]['loss'] = loss
        scores = self.temp_score[dtype]
        scores = sorted(scores.items(), key=functools.cmp_to_key(self.metrics.metric_item_cmp), reverse=False)
        scores = {item[0]: float(f"{item[1]:.4f}") for item in scores}
        return deepcopy(scores)

    def finish_epoch(self):
        for dtype in ['train', 'val', 'test']:
            self.wandb_run.log({dtype: self.temp_score[dtype]}, step=self.epoch)
        self.epoch += 1
        temp_score = self.temp_score
        self.temp_score = None

        # set the last epoch result as the best result when early stopper is not set.
        if self.early_stopper is None:
            for dtype in ['train', 'val', 'test']:
                self.final_scores[dtype] = temp_score[dtype]
                self.wandb_run.summary.update({dtype: temp_score[dtype]})
        else:
            early_stopper_state = self.early_stopper.early_stop_check(
                sum(temp_score['val'].values()) - temp_score['val']['loss'])
            if early_stopper_state == 'break':
                return True
            if early_stopper_state == 'better':  # find better result
                for dtype in ['train', 'val', 'test']:
                    self.final_scores[dtype] = temp_score[dtype]
                    self.wandb_run.summary.update({dtype: temp_score[dtype]})
            return False

    def get_final_scores(self):
        return deepcopy(self.final_scores)

    def save(self):
        save_result = dict()
        for dtype in ['train', 'val', 'test']:
            score = self.final_scores[dtype]
            save_result[dtype] = {**score}
        pk.dump(save_result, open(self.path, 'wb'))

    def info_epoch(self):
        end_time = time.time()
        self.logger.info(f"Epoch{self.epoch}: ")
        self.logger.info(f"Time cost is {end_time - self.start_time:.4f}s")
        for dtype in ['train', 'val', 'test']:
            s = []
            for metric in self.metric_names:
                s.append(f'{metric}:{self.temp_score[dtype][metric]:.4f}')
            self.logger.info(f'{dtype}: ' + '\t'.join(s))


class Metrics(object):
    def __init__(self, k_list=None):
        super().__init__()
        if k_list is None:
            k_list = [10, 50, 100]
        self.k_list = k_list

    def get_metric_names(self):
        metrics = []
        for top_k in self.k_list:
            metrics.extend([
                f'recall_{top_k}', f'map_{top_k}',
            ])
        metrics.sort(key=functools.cmp_to_key(self.metric_item_cmp), reverse=False)
        return metrics

    def metric_item_cmp(self, x, y):
        if type(x) == tuple:
            x, y = x[0], y[0]
        x = x.split('_')
        y = y.split('_')
        if x[0] > y[0]:
            return 1
        elif x[0] < y[0]:
            return -1
        else:
            return int(x[1]) - int(y[1])

    def get_metrics(self, y_pred, y_true, mean_reduce=True):
        """
            Args:
                y_true: tensor (samples_num, num_items)
                y_pred: tensor (samples_num, num_items)
            Returns:
                scores: dict
        """

        result = {}
        for top_k in self.k_list:
            result.update({
                f'recall_{top_k}': self.recall_score(y_pred, y_true, top_k=top_k, mean_reduce=mean_reduce),
                f'map_{top_k}': self.map(y_pred, y_true, top_k=top_k, mean_reduce=mean_reduce),
            })
        result = sorted(result.items(), key=functools.cmp_to_key(self.metric_item_cmp), reverse=False)
        result = {item[0]: float(f"{item[1]:.4f}") for item in result}
        return result

    def recall_score(self, y_pred, y_true, top_k, mean_reduce):
        """
        Args:
            y_true (Tensor): shape (batch_size, num_items)
            y_pred (Tensor): shape (batch_size, num_items)
            top_k (int):
        Returns:
            output (float)
        """
        _, predict_indices = y_pred.topk(k=top_k)
        predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                                 value=1).long(), y_true.long()
        tp, t = ((predict == truth) & (truth == 1)).sum(-1), truth.sum(-1)
        recall = tp.float() / t.float()
        recall[t == 0] = 1
        recall = recall.mean().item() if mean_reduce else recall.sum().item()
        return recall

    def dcg(self, y_true: torch.Tensor, y_pred: torch.Tensor, top_k):
        """
        Args:
            y_true: (batch_size, num_items)
            y_pred: (batch_size, num_items)
            top_k (int):

        Returns:

        """
        _, predict_indices = y_pred.topk(k=top_k)
        gain = y_true.gather(-1, predict_indices)  # (batch_size, top_k)
        return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(
            -1)  # (batch_size,)

    def ndcg_score(self, y_pred, y_true, top_k, mean_reduce):
        dcg_score = self.dcg(y_true, y_pred, top_k)
        idcg_score = self.dcg(y_true, y_true, top_k)
        ndcg = dcg_score / idcg_score
        ndcg[idcg_score == 0] = 1
        ndcg = ndcg.mean().item() if mean_reduce else ndcg.sum().item()
        return ndcg

    def map(self, y_pred, y_true, top_k, mean_reduce):
        """
        Args:
            y_true: (batch_size, num_items)
            y_pred: (batch_size, num_items)
            top_k (int):

        Returns: float
        """
        _, predict_indices = y_pred.topk(k=top_k)
        # [batch,num_items]
        topk_labels = torch.gather(y_true, -1, predict_indices)
        hit_cnt = topk_labels.sum(dim=1)
        cumsum_topk_labels = torch.cumsum(topk_labels, dim=1)
        cumsum_topk_labels = topk_labels * cumsum_topk_labels
        map = torch.sum(cumsum_topk_labels / torch.arange(1, top_k + 1,device=y_pred.device)[None, :], dim=1) / hit_cnt
        map[hit_cnt == 0] = 0
        label_cnt = y_true.sum(-1)
        map[label_cnt == 0] = 1
        assert torch.sum(label_cnt == 0) == 0
        map = map.mean().item() if mean_reduce else map.sum().item()
        return map

    def PHR(self, y_pred, y_true, top_k, mean_reduce):
        """
        Args:
            y_true (Tensor): shape (batch_size, num_items)
            y_pred (Tensor): shape (batch_size, num_items)
            top_k (int):
        Returns:
            output (float)
        """
        _, predict_indices = y_pred.topk(k=top_k)
        predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                                 value=1).long(), y_true.long()
        hit_num = torch.mul(predict, truth).sum(dim=1).nonzero().shape[0]
        hit_num += torch.sum(truth.sum(-1) == 0)
        hit_num = hit_num / truth.shape[0] if mean_reduce else hit_num
        return hit_num


def set_config(args):
    if args.embedding_module == 'concat':
        args.emb_dim = 2 * args.emb_dim
    param = deepcopy(vars(args))
    param['prefix'] = f'{args.prefix}_{args.dataset}_ours_{args.message_generator}'
    param['model_path'] = f"/home/luxd/r-project/CTDG_Cas/backbone/saved_models/{param['prefix']}"
    param['result_path'] = f"/home/luxd/r-project/CTDG_Cas/backbone/results/{param['prefix']}"
    param['log_path'] = f"/home/luxd/r-project/CTDG_Cas/backbone/log/{param['prefix']}.log"
    return param
