import numpy as np
import torch
import torch.nn as nn
from typing import Union, List
from ASDIP.graph_sampler import WalkSampler, NeighborSampler
from ASDIP.aggregator import get_aggregator
from ASDIP.merger import get_merger
from ASDIP.time_encoder import TrendEncoder


def output(name, tensor):
    print(f'{name}:', torch.mean(torch.abs(tensor)), torch.std(torch.abs(tensor)))
    print(tensor[:5, :5])


class PadEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim, device, all_zero=False):
        super(PadEmbedding, self).__init__()
        self.all_zero = all_zero
        self.embedding_num = num_embedding
        self.embedding_dim = embedding_dim
        self.device = device
        if self.all_zero is not True:
            self.embeddings = nn.Embedding(num_embedding + 1, embedding_dim, padding_idx=-1)

    def forward(self, tgt):
        if self.all_zero:
            return torch.zeros((*tgt.shape, self.embedding_dim), device=self.device)
        new_tgt = tgt.clone()
        new_tgt = new_tgt.reshape(-1)
        new_tgt[new_tgt == -1] += self.embedding_num + 1
        emb = self.embeddings(new_tgt)
        return emb.reshape(*tgt.shape, -1)


class CasEmbedding(nn.Module):
    def __init__(self, emb_type, dim, user_emb, cas_num, neighbor_sampler: NeighborSampler, device):
        super().__init__()
        self.emb_type = emb_type
        self.neighbor_sampler = neighbor_sampler
        self.device = device
        if self.emb_type == 'zero':
            self.embedding = PadEmbedding(num_embedding=cas_num, embedding_dim=dim, device=self.device, all_zero=True)
        elif self.emb_type == 'aggregate':
            self.embedding = user_emb
        else:
            raise ValueError("Not Implemented Cas Embedding Type")

    def forward(self, tgt, times):
        if self.emb_type == 'zero':
            return self.embedding(tgt)
        elif self.emb_type == 'aggregate':
            concat_tgt = torch.stack([tgt, times], dim=1)
            unique_concat_tgt, reverse_indices = torch.unique(concat_tgt, dim=0, return_inverse=True)
            unique_tgt, unique_times = unique_concat_tgt[:, 0].to(torch.long), unique_concat_tgt[:, 1]
            sampled_neighbors, lengths = self.neighbor_sampler.get_cas_neighbors(
                nodes=unique_tgt.cpu().numpy().reshape(-1),
                times=unique_times.cpu().numpy().reshape(-1)
            )
            sampled_neighbors = torch.tensor(sampled_neighbors, device=self.device)
            lengths = torch.tensor(lengths, device=self.device)
            embedding = torch.sum(self.embedding(sampled_neighbors), dim=1)
            embedding = embedding / (lengths[:, None] + 1e-9)
            return embedding[reverse_indices].reshape(*tgt.shape, -1)


class ASDIP(nn.Module):
    def __init__(self, dim, device, dropout, user_num, walk_samplers: List[WalkSampler], sample_num, use_time,
                 num_layer, predict_time_delta, time_dim, time_max_length, logger, path_encoder, cas_num, walk_length,
                 view_merger, use_self_loss, neighbor_sampler, cas_emb_type):
        super(ASDIP, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.user_emb = PadEmbedding(num_embedding=user_num, embedding_dim=dim, device=device)
        self.cas_emb = CasEmbedding(emb_type=cas_emb_type, dim=dim, user_emb=self.user_emb, cas_num=cas_num,
                                    neighbor_sampler=neighbor_sampler, device=device)
        self.user_num = user_num
        self.num_layer = num_layer
        self.logger = logger

        self.time_dim = 0
        self.dim = dim
        if use_time:
            step_length = int(predict_time_delta // 4)
            max_length = step_length * 4 * time_max_length
            self.time_encoder = TrendEncoder(time_dim, device, step_length, max_length)
            self.time_dim = time_dim
        self.aggregator = get_aggregator(user_emb=self.user_emb, walk_samplers=walk_samplers,
                                         sample_num=sample_num, dim=dim, dropout=dropout, device=device,
                                         path_encoder=path_encoder, cas_emb=self.cas_emb, logger=logger,
                                         walk_length=walk_length, view_merger=view_merger, use_self_loss=use_self_loss)
        self.merger = get_merger(dim=dim, mask_after=True, num_layer=num_layer, device=device, dropout=dropout)

        self.query = nn.Linear(in_features=dim, out_features=1, bias=False)
        self.decoder = nn.Linear(in_features=dim + self.time_dim, out_features=user_num)

        self.use_time = use_time
        self.graphs = walk_samplers
        self.sample_num = sample_num

    def get_padding_mask(self, x, length):
        batch, step, _ = x.shape
        mask = torch.arange(1, step + 1)[None, :]
        mask = ~(mask <= torch.tensor(length)[:, None])
        return mask

    def forward(self, users, length, interact_times, predict_times):
        batch, max_seq_length = users.shape
        static_user_embs = self.user_emb(users)
        key_pad_mask = self.get_padding_mask(static_user_embs, length).to(self.device)
        static_user_embs = self.dropout(static_user_embs)

        user_mask = ~(key_pad_mask.reshape(-1))
        dynamic_user_embs = torch.zeros((torch.numel(users), self.dim), device=self.device)
        dynamic_user_embs[user_mask] = self.aggregator(nodes=users.reshape(-1)[user_mask],
                                                       times=interact_times.reshape(-1)[user_mask],
                                                       predict_times=torch.repeat_interleave(
                                                           predict_times, max_seq_length)[user_mask])
        dynamic_user_embs = dynamic_user_embs.reshape(batch, max_seq_length, -1)

        unified_user_embs = self.merger(static_user_embs, dynamic_user_embs, key_pad_mask)
        matrix = self.query(unified_user_embs)
        matrix[key_pad_mask] = -1e9
        matrix = torch.softmax(matrix, dim=1)
        cascade_user_embs = torch.sum(matrix * unified_user_embs, dim=1)
        if self.use_time:
            cascade_time_embs = self.dropout(self.time_encoder(interact_times, length, predict_times))
            cascade_embs = torch.cat([cascade_user_embs, cascade_time_embs], dim=1)
        else:
            cascade_embs = cascade_user_embs
        pred = self.decoder(cascade_embs)
        users[key_pad_mask] = self.user_num
        occ_mask = torch.zeros((users.shape[0], self.user_num + 1), device=self.device)
        occ_mask = torch.scatter(occ_mask, dim=1, index=users, value=-1e9)
        pred = pred + occ_mask[:, :-1]
        return pred


def get_model(dim, device, use_time_emb, dropout, user_num, walk_samplers, sample_num, num_layer,
              predict_time_delta, time_dim, time_max_length, logger, path_encoder, cas_num, walk_length,
              view_merger, use_self_loss, neighbor_sampler, cas_emb_type):
    return ASDIP(dim=dim, device=device, dropout=dropout, user_num=user_num, walk_samplers=walk_samplers,
                 sample_num=sample_num, use_time=use_time_emb, num_layer=num_layer,
                 predict_time_delta=predict_time_delta, time_dim=time_dim,
                 time_max_length=time_max_length, logger=logger, path_encoder=path_encoder, cas_num=cas_num,
                 walk_length=walk_length, view_merger=view_merger,
                 use_self_loss=use_self_loss, neighbor_sampler=neighbor_sampler, cas_emb_type=cas_emb_type)
