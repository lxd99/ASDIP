import logging
import pickle
import sys
import os
import yaml
from utils.my_utils import get_logger
from utils.dataloader import get_seq_data
import json

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "120"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# If you want to use wandb, please comment out the following two lines
os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DISABLED'] = 'true'

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from ASDIP.ASDIP_model import get_model, ASDIP
import argparse
from utils.Metric import EarlyStopMonitor, MetricManager
import pickle as pk
from utils.my_utils import setup_seed, setup_thread, str2bool, merge_args
from ASDIP.graph_sampler import WalkSampler, NeighborSampler
import time


class myDataSet(Dataset):
    def __init__(self, targets, seqs, labels, times, predict_times, max_seq):
        self.len = len(seqs)
        self.seqs = seqs
        self.labels = labels
        self.targets = targets
        self.times = times
        self.predict_times = predict_times
        self.max_len = max_seq

    def __getitem__(self, item):
        seq_data, time_data = self.seqs[item], self.times[item]
        if len(seq_data) > self.max_len:
            seq_data = seq_data[-self.max_len:]
            time_data = time_data[-self.max_len:]
        return self.targets[item], torch.tensor(seq_data), self.labels[item], len(seq_data), \
            torch.tensor(time_data), self.predict_times[item]

    def __len__(self):
        return self.len


def seq_collate(batch):
    targets, datas, labels, valid_lengths, times, predict_times = zip(*batch)
    datas = pad_sequence(datas, batch_first=True).to(device)
    times = pad_sequence(times, batch_first=True).to(device)
    targets, valid_lengths, predict_times = np.array(targets), np.array(valid_lengths), torch.tensor(predict_times,
                                                                                                     device=device)
    label_indices = [[row, col] for row, x in enumerate(labels) for col in x]
    label_tensor = torch.sparse_coo_tensor(torch.tensor(label_indices).t(), [1] * len(label_indices),
                                           (len(labels), user_num), device=device).to_dense()
    return targets, datas, label_tensor, valid_lengths, times, predict_times


def time_trans(x):
    if x == 'months':
        return 30 * 86400
    elif x == 'days':
        return 86400
    elif x == 'years':
        return 365 * 30 * 86400
    else:
        raise ValueError("Not implemented time scale")


def gen_dataset(dataset, train_data_split, max_seq, logger):
    data = pd.read_csv(f'data/dynamic/{dataset}.csv')
    all_labels = pk.load(open(f'data/dynamic/{dataset}_label.pkl', 'rb'))
    predict_time_delta = time_trans(all_labels['predict_unit']) * all_labels["predict_unit_num"]
    # predict_start.value // 10 ** 9
    limit_times = all_labels['train_time'] + [all_labels['val_time'], all_labels['test_time']]
    limit_times = (pd.to_datetime(limit_times).values.view(np.int64) // 10 ** 9).astype(np.float32)
    user_num = max(data['dst']) + 1
    cascade_num = max(data['cas']) + 1
    seq_data = get_seq_data(data, all_labels, logger, train_data_split)
    mydata = dict()
    for dtype in ['train', 'val', 'test']:
        m_seq_data = seq_data[dtype]
        mydata[dtype] = myDataSet(m_seq_data['id'], m_seq_data['cascade'], m_seq_data['label'], m_seq_data['time'],
                                  m_seq_data['predict_time'], max_seq)

    return data, mydata['train'], mydata['val'], mydata['test'], user_num, cascade_num, predict_time_delta, limit_times


def test_seq(model: ASDIP, loader, device, metric: MetricManager, dtype, walk_samplers):
    model.eval()
    for walk_sampler in walk_samplers:
        walk_sampler.set_state('eval')
    for target, x, y, length, time, predict_time in tqdm(loader):
        with torch.no_grad():
            pred = model(x, length, time, predict_time)
            metric.update(target=target, pred=pred, label=y, dtype=dtype)
    return metric.calculate_metric(dtype=dtype)


def train_seq(train, val, test, num_epoch, model: ASDIP, device, logger, model_path,
              metric: MetricManager, runs, use_self_loss, patience, lam, lr, walk_samplers):
    early_stopper = EarlyStopMonitor(max_round=patience, higher_better=True, tolerance=1e-10, save_path=model_path,
                                     model=model, run=my_seed)
    metric.init_run(early_stopper)
    # metric.watch(model)
    optim = Adam(params=model.parameters(), lr=lr)
    for epoch in range(num_epoch):
        metric.init_epoch()
        model.train()
        for walk_sampler in walk_samplers:
            walk_sampler.set_state('train')
        train_loss, batch = 0, 0
        for target, x, y, length, time, predict_time in tqdm(train):
            pred = model(x, length, time, predict_time)
            metric.update(target=target, pred=pred.detach(), label=y,
                          dtype='train')
            loss = nn.functional.binary_cross_entropy_with_logits(pred, y.float())
            if use_self_loss:
                self_score, self_label = model.aggregator.loss_data
                if self_score is not None:
                    self_loss = lam * nn.functional.cross_entropy(self_score, self_label)
                    loss = loss + self_loss
            train_loss += loss.item()
            batch += 1
            optim.zero_grad()
            loss.backward()
            optim.step()
        train_metric = metric.calculate_metric('train', loss=train_loss / batch)
        val_metric = test_seq(model, val, device, metric, 'val', walk_samplers)
        test_metric = test_seq(model, test, device, metric, 'test', walk_samplers)
        metric.info_epoch()
        if metric.finish_epoch():
            logger.info('No improvement over {} epochs, stop training'.format(metric.early_stopper.max_round))
            break
        else:
            ...
    logger.info(f'Loading the best model at epoch {metric.early_stopper.best_epoch}')
    model.load_state_dict(torch.load(f"{model_path}_{runs}.pth"))
    logger.info(f'Loaded the best model at epoch {metric.early_stopper.best_epoch} for inference')
    metric.init_epoch()
    best_score = test_seq(model, test, device, metric, 'test', walk_samplers)
    logger.info(f'Runs {runs}: {best_score}')
    metric.finish_run()
    return best_score


def get_args(config_path) -> argparse.Namespace:
    parser = argparse.ArgumentParser('ASDIP Training')
    parser.add_argument('--prefix', type=str, help='prefix', default='test')
    parser.add_argument('--dataset', type=str, help='Dataset name ', default='memetracker')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n_runs', type=int, default=3, help='the number of runs')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--use_time', type=str2bool, default=True,
                        help='whether to use the time of the user sequence of a cascade')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--model', type=str, default='ASDIP', choices=['ASDIP'])
    parser.add_argument('--max_seq', type=int, default=200)
    parser.add_argument('--train_data_split', type=str, choices=['last', 'all'], default='all')
    parser.add_argument('--sample_num', type=int, default=30)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--time_dim', type=int, default=32)
    parser.add_argument('--time_max_len', type=int, default=3)
    parser.add_argument('--causal', type=str, nargs="+", default=['none'], help='causal walk type')
    parser.add_argument('--use_saved_walk', type=str2bool, default=True, help='whether to use the saved random walks')
    parser.add_argument('--path_encoder', type=str, choices=['mean','rnn',  'mlp', 'conv'], default='mlp',
                        help='encoder type for path node')
    parser.add_argument('--view_merger', type=str, choices=['mlp', 'mlp_gate'],
                        default='mlp_gate',
                        help='the merge type of multi-view rep')
    parser.add_argument('--use_self_loss', type=str2bool, default=True, help='whether to add self-loss')
    parser.add_argument('--cas_emb_type', type=str, default='aggregate', choices=['zero', 'aggregate'],
                        help='the cascade embedding type')
    ################### args which are not the same in different datasets ###############
    parser.add_argument('--lam', type=float, help='the weight for the self-supervised loss')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--walk_length', type=int, help='the length of the sampled walk')
    args = parser.parse_args()
    args = merge_args(args, yaml.load(open(config_path, 'r'), yaml.FullLoader)[args.dataset])
    return args


if __name__ == '__main__':
    args = get_args(config_path='config/ours.yml')
    device = torch.device(f'cuda:{args.gpu}')
    with open(f"saved_models/{args.prefix}_{args.dataset}_{args.model}.json", "w") as f:
        f.write(json.dumps(vars(args)))
    logging.getLogger('numba').setLevel(logging.WARNING)
    logger = get_logger(f'log/{args.prefix}_{args.dataset}_{args.model}.log')
    logger.info(args)

    data, train, val, test, user_num, cas_num, predict_time_delta, limit_times = gen_dataset(dataset=args.dataset,
                                                                                             train_data_split=args.train_data_split,
                                                                                             max_seq=args.max_seq,
                                                                                             logger=logger)

    walk_samplers = []
    for causal in args.causal:
        walk_sampler = WalkSampler(users=list(data['dst']), cascades=list(data['cas']), times=list(data['time']),
                                   user_num=user_num, cascade_num=cas_num, device=device, causal=causal,
                                   sample_num=args.sample_num, limit_times=limit_times, walk_length=args.walk_length,
                                   dataset=args.dataset, use_saved=args.use_saved_walk)
        walk_samplers.append(walk_sampler)
    neighbor_sampler = NeighborSampler(users=list(data['dst']), cascades=list(data['cas']), times=list(data['time']),
                                       user_num=user_num, cascade_num=cas_num, max_neighbor_num=args.max_seq // 4)
    logger.info(f'Dataset is {args.dataset}')
    # set your wandb configure when used wandb
    my_wandb_config = {'name':'your_wandb_run_name'}
    model_path = f'saved_models/{args.prefix}_{args.dataset}_{args.model}'
    metric = MetricManager(path=f'results/{args.prefix}_{args.dataset}_{args.model}', logger=logger,
                           wandb_config=my_wandb_config)

    setup_thread()
    for my_seed in range(args.n_runs):
        run_start = time.time()
        logger.info(f'--------Begin Run with Seed {my_seed}--------------')
        setup_seed(my_seed, torch_only_deterministic=True)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        for walk_sampler in walk_samplers:
            walk_sampler.set_state('train')
            walk_sampler.set_seed(my_seed)
            if walk_sampler.use_saved:
                walk_sampler.load_saved_data()
        # dataloader多线程的时候需要固定seed
        train_loader = DataLoader(train, args.batch, shuffle=True, collate_fn=seq_collate)
        val_loader = DataLoader(val, args.batch, shuffle=True, collate_fn=seq_collate)
        test_loader = DataLoader(test, args.batch, shuffle=True, collate_fn=seq_collate)
        model = get_model(dim=args.dim, device=device, use_time_emb=args.use_time,
                          dropout=args.dropout, user_num=user_num, walk_samplers=walk_samplers,
                          sample_num=args.sample_num, num_layer=args.num_layer, predict_time_delta=predict_time_delta,
                          time_dim=args.time_dim,
                          time_max_length=args.time_max_len, logger=logger, path_encoder=args.path_encoder,
                          cas_num=cas_num, walk_length=args.walk_length,
                          view_merger=args.view_merger, use_self_loss=args.use_self_loss,
                          neighbor_sampler=neighbor_sampler, cas_emb_type=args.cas_emb_type).to(device)
        _ = train_seq(num_epoch=args.epoch, model=model, device=device, train=train_loader,
                      val=val_loader, test=test_loader, logger=logger,
                      model_path=model_path, metric=metric, runs=my_seed, use_self_loss=args.use_self_loss,
                      patience=args.patience, lam=args.lam, lr=args.lr, walk_samplers=walk_samplers)
        run_end = time.time()
        logger.info(f'Run {my_seed + 1} cost {run_end - run_start}s')

    avg_metric = metric.finish()
    logger.info(avg_metric)
