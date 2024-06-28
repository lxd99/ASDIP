import torch.nn as nn
import torch
from ASDIP.graph_sampler import WalkSampler
from ASDIP.walk_encoder import get_walk_encoder
from typing import Union


class WalkAggregator(nn.Module):
    def __init__(self, user_emb, sample_num, dim, dropout, device, path_encoder, cas_emb, logger,
                 walk_length):
        super(WalkAggregator, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.sample_num = sample_num
        self.walk_length = walk_length
        self.logger = logger
        self.user_emb = user_emb
        self.path_encoder = get_walk_encoder(encoder_type=path_encoder, user_emb=user_emb, cas_emb=cas_emb,
                                             dim=dim, device=device, walk_length=walk_length, dropout=0.1,
                                             walk_num=sample_num)

    def get_context_emb(self, nodes, times, predict_times, walk_sampler: WalkSampler):
        """
        @param nodes: tensor of shape [batch]
        @param times: tensor of shape [batch]
        @param predict_times: tensor of shape [batch]
        @param walk_sampler: walk sampler
        @return: (path_emb,index_emb), shape of path_emb is [batch,dim],
        index_emb is Union[Tensor], shape is [batch,dim]
        """
        # [batch,sample_num,walk_length]
        walk_nodes = walk_sampler.get_walks(
            nodes.reshape(-1).cpu().numpy(),
            times.reshape(-1).cpu().numpy(),
            predict_times.reshape(-1).cpu().numpy())
        # [batch*sample_num,dim]
        path_emb = self.path_encoder.get_path_emb(walk_nodes=walk_nodes,
                                                  predict_times=predict_times)
        path_emb = path_emb.reshape(-1, self.sample_num, self.dim)
        path_emb = torch.sum(path_emb, dim=1)
        return self.dropout(path_emb)


class ViewAggregator(nn.Module):
    def __init__(self, user_emb, walk_samplers, sample_num, dim, dropout, device, path_encoder,
                 cas_emb, logger, walk_length, merge_type, use_self_loss):
        super(ViewAggregator, self).__init__()
        self.logger = logger
        self.view_num = len(walk_samplers)
        self.walk_samplers = walk_samplers
        self.merge_type = merge_type
        self.device = device
        self.use_self_loss = use_self_loss
        self.dim = dim
        self.walk_num = sample_num
        self.walk_length = walk_length
        self.walk_aggregators = nn.ModuleList([
            WalkAggregator(user_emb=user_emb, sample_num=sample_num, dim=dim, dropout=dropout, device=device,
                           path_encoder=path_encoder, cas_emb=cas_emb, logger=logger, walk_length=walk_length)
            for _ in walk_samplers])

        if self.view_num > 1:
            if self.merge_type == 'mlp':
                self.scale_merger = nn.Sequential(nn.Linear(dim * self.view_num, dim), nn.ReLU(),
                                                  nn.Linear(dim, dim))
            elif self.merge_type == 'mlp_gate':
                self.scale_merger = nn.Sequential(nn.Linear(dim * self.view_num, dim * self.view_num), nn.ReLU(),
                                                  nn.Linear(dim * self.view_num, dim * self.view_num))
            else:
                raise ValueError("Not Implemented Merge Type!")

            if self.use_self_loss:
                self.score_projector = nn.Linear(dim, dim, bias=False)

    def get_ctx_emb(self, nodes, times, predict_times):
        ctx_embeddings = []
        for i in range(self.view_num):
            # [batch,step,dim]
            ctx_emb = self.walk_aggregators[i].get_context_emb(nodes, times, predict_times,
                                                               self.walk_samplers[i])
            ctx_embeddings.append(ctx_emb)

        return torch.concat(ctx_embeddings, dim=1)

    def calculate_self_loss(self, ctx_emb):
        """
        @param ctx_emb: tensor of shape [batch,num_view*dim]
        @return:
        """
        batch, _ = ctx_emb.shape
        ctx_emb = ctx_emb.reshape(batch, self.view_num, self.dim)
        score_list = []
        label_list = []
        for i in range(self.view_num):
            for j in range(i + 1, self.view_num):
                ctx_emb_view_1 = ctx_emb[:, i, :]
                ctx_emb_view_2 = ctx_emb[:, j, :]
                score_list.append(
                    torch.matmul(self.score_projector(ctx_emb_view_1), self.score_projector(ctx_emb_view_2).T))
                label_list.append(torch.arange(batch, device=self.device))
        scores = torch.cat(score_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        self.loss_data = [scores, labels]

    def forward(self, nodes, times, predict_times):
        """
        @param nodes: tensor of shape [batch]
        @param times: tensor of shape [batch]
        @param predict_times: tensor of shape [batch]
        @return ctx_emb: tensor of shape [batch,dim]
        """
        ctx_emb = self.get_ctx_emb(nodes, times, predict_times)
        if self.view_num > 1:
            if self.merge_type == 'mlp':
                combined_ctx_emb = self.scale_merger(ctx_emb)
            elif self.merge_type == 'mlp_gate':
                gate = self.scale_merger(ctx_emb)
                gate = torch.softmax(gate.reshape(-1, self.view_num, self.dim), dim=1)
                combined_ctx_emb = ctx_emb.reshape(-1, self.view_num, self.dim)
                combined_ctx_emb = torch.sum(gate * combined_ctx_emb, dim=1)
            else:
                raise ValueError("Not Implemented merge type")
            if self.use_self_loss and self.training:
                self.calculate_self_loss(ctx_emb)
        else:
            combined_ctx_emb = ctx_emb
        return combined_ctx_emb


def get_aggregator(user_emb, walk_samplers, sample_num, dim, dropout, device, path_encoder,
                   cas_emb, logger, walk_length, view_merger, use_self_loss) -> ViewAggregator:
    return ViewAggregator(user_emb=user_emb, walk_samplers=walk_samplers, sample_num=sample_num, dim=dim,
                          dropout=dropout, device=device, path_encoder=path_encoder, cas_emb=cas_emb, logger=logger,
                          walk_length=walk_length, merge_type=view_merger, use_self_loss=use_self_loss)
