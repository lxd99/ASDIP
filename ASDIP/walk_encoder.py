import numpy as np
import pandas as pd
import torch.nn as nn
import torch


class WalkEncoder(nn.Module):
    def __init__(self, user_emb: nn.Module, cas_emb: nn.Module, device, dim, walk_length,
                 path_drop, node_dim, walk_num):
        super(WalkEncoder, self).__init__()
        self.user_emb = user_emb
        self.cas_emb = cas_emb
        self.dim = dim
        self.node_dim = node_dim
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.dropout = nn.Dropout(p=path_drop)
        self.device = device

    def get_node_emb(self, walk_nodes, predict_times):
        walk_nodes_emb = []
        for i in range(1, self.walk_length):
            if i % 2 == 1:
                walk_nodes_emb.append(self.cas_emb(walk_nodes[:, i], predict_times[:, i]))
            else:
                walk_nodes_emb.append(self.user_emb(walk_nodes[:, i]))

        # [batch,step_num-1/step,dim]
        walk_nodes_emb = torch.stack(walk_nodes_emb, dim=1)
        return walk_nodes_emb

    def agg_path_info(self, walk_nodes, predict_times):
        ...

    def get_path_emb(self, walk_nodes, predict_times):
        """
        @param walk_nodes: tensor of shape [batch,walk_num,step]
        @param predict_times: tensor of shape [batch]
        @return: a tensor of shape [batch*walk_num,dim]
        """
        batch, walk_num, step = walk_nodes.shape
        walk_nodes = walk_nodes.reshape(batch * walk_num, step)
        predict_times = predict_times.repeat_interleave(walk_num * step).reshape(batch * walk_num, step)
        path_emb = self.agg_path_info(walk_nodes, predict_times)
        return path_emb


class MeanEncoder(WalkEncoder):
    def agg_path_info(self, walk_nodes, predict_times):
        # [batch,step_num,dim]
        walk_node_emb = self.dropout(self.get_node_emb(walk_nodes, predict_times))
        walk_node_emb = torch.mean(walk_node_emb, dim=1)
        return walk_node_emb


class RNNEncoder(WalkEncoder):
    def __init__(self, user_emb, cas_emb, device, dim, walk_length, path_drop, node_dim,
                 walk_num):
        super(RNNEncoder, self).__init__(user_emb, cas_emb, device, dim, walk_length, path_drop,
                                         node_dim, walk_num)
        self.node_aggregator = nn.LSTM(input_size=node_dim, hidden_size=dim, batch_first=True)

    def agg_path_info(self, walk_nodes, predict_times):
        # [batch,step_num,dim]
        walk_node_emb = self.dropout(self.get_node_emb(walk_nodes, predict_times))
        batch, step_num, dim = walk_node_emb.shape
        _, (walk_node_emb, _) = self.node_aggregator(walk_node_emb)
        return walk_node_emb.squeeze(dim=0)


class MLPEncoder(WalkEncoder):
    def __init__(self, user_emb, cas_emb, device, dim, walk_length, path_drop, node_dim,
                 walk_num):
        super(MLPEncoder, self).__init__(user_emb, cas_emb, device, dim, walk_length, path_drop,
                                         node_dim, walk_num)
        token_dim = walk_length - 1
        self.mlp = nn.Linear(token_dim, 1)

    def agg_path_info(self, walk_nodes, predict_times):
        # [batch,step_num,dim]
        walk_node_emb = self.dropout(self.get_node_emb(walk_nodes, predict_times))
        walk_node_emb = self.mlp(walk_node_emb.transpose(1, 2)).squeeze(dim=2)
        return walk_node_emb


class ConvEncoder(WalkEncoder):
    def __init__(self, user_emb, cas_emb, device, dim, walk_length, path_drop, node_dim,
                 walk_num):
        super(ConvEncoder, self).__init__(user_emb, cas_emb, device, dim, walk_length, path_drop,
                                          node_dim, walk_num)
        self.conv = nn.Conv1d(in_channels=node_dim, out_channels=dim, kernel_size=walk_length - 1)

    def agg_path_info(self, walk_nodes, predict_times):
        walk_node_emb = self.dropout(self.get_node_emb(walk_nodes, predict_times))
        return self.conv(walk_node_emb.transpose(1, 2)).squeeze(2)


def get_walk_encoder(encoder_type, user_emb, cas_emb, dim, device,
                     walk_length, dropout, walk_num) -> WalkEncoder:
    if encoder_type == 'mean':
        return MeanEncoder(user_emb, cas_emb, device, dim, walk_length, dropout, dim, walk_num)
    elif encoder_type == 'rnn':
        return RNNEncoder(user_emb, cas_emb, device, dim, walk_length, dropout, dim,
                          walk_num)
    elif encoder_type == 'mlp':
        return MLPEncoder(user_emb, cas_emb, device, dim, walk_length, dropout, dim, walk_num)
    elif encoder_type == 'conv':
        return ConvEncoder(user_emb, cas_emb, device, dim, walk_length, dropout, dim, walk_num)
    else:
        raise ValueError("Not implemented path aggregator")
