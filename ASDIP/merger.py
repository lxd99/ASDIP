import torch.nn as nn
import torch

class Merger(nn.Module):
    def __init__(self, dim, mask_after, num_layer, device):
        super(Merger, self).__init__()
        self.dim = dim
        self.mask_after = mask_after
        self.device = device
        self.num_layer = num_layer

    def merge(self, node_emb, ctx_emb, key_pad_mask, att_mask):
        ...

    def forward(self, node_emb, ctx_emb, key_pad_mask):
        if self.mask_after:
            att_mask = torch.triu(torch.ones(node_emb.shape[1], node_emb.shape[1]), diagonal=1).to(dtype=torch.bool,
                                                                                                   device=self.device)
        else:
            att_mask = torch.zeros(node_emb.shape[1], node_emb.shape[1]).to(dtype=torch.bool, device=self.device)
        return self.merge(node_emb, ctx_emb, key_pad_mask, att_mask)


class ContextMerger(Merger):
    def __init__(self, dim, mask_after, num_layer, device, dropout):
        super(ContextMerger, self).__init__(dim, mask_after, num_layer, device)
        merger_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=2,
                                                  batch_first=True, dropout=dropout)  # tgt会做self-attention,mem不会
        self.merger = nn.TransformerEncoder(encoder_layer=merger_layer, num_layers=num_layer)

    def merge(self, node_emb, ctx_emb, key_pad_mask, att_mask):
        x_emb = self.merger(src=ctx_emb, mask=att_mask, src_key_padding_mask=key_pad_mask)
        return x_emb


def get_merger(dim, mask_after, num_layer, device, dropout):
    return ContextMerger(dim, mask_after, num_layer, device, dropout)
