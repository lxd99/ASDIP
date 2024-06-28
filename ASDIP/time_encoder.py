import pandas as pd
import torch
import numpy as np
import torch.nn as nn


class TGATEncoder(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TGATEncoder, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=1)

        # output has shape [batch_size, dimension]
        output = torch.cos(self.w(t))

        return output


class EmbedEncoder(torch.nn.Module):
    def __init__(self, dimension, max_time, time_num):
        super(EmbedEncoder, self).__init__()
        self.max_time = max_time
        self.time_num = time_num
        self.emb = nn.Embedding(time_num, dimension)

    def forward(self, t):
        t = (t / self.max_time * (self.time_num - 1)).to(torch.long)
        return self.emb(t)


class FixTGATEncoder(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(FixTGATEncoder, self).__init__()

        self.dimension = dimension
        alpha, beta = np.sqrt(dimension), np.sqrt(dimension)
        w = torch.tensor([np.power(alpha, -i / beta) for i in range(self.dimension)], dtype=torch.float)
        self.w = nn.Parameter(w, requires_grad=False)

    def forward(self, t):
        # t has shape [batch_size]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=1)

        # output has shape [batch_size, dimension]
        output = torch.cos(self.w[None, :] * t)

        return output


class SemanticEncoder(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension, device):
        super(SemanticEncoder, self).__init__()
        self.dimension = dimension
        self.week_emb = nn.Embedding(7, dimension)
        self.month_emb = nn.Embedding(12, dimension)
        self.day_emb = nn.Embedding(31, dimension)
        self.device = device

    def forward(self, t):
        # t has shape [batch_size]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        new_t = t.cpu().numpy()
        new_t = pd.to_datetime(new_t, unit='s')
        week, day, month = new_t.weekday.values, new_t.day.values, new_t.month.values
        # print(new_t[:5])
        # print(week[:5], day[:5], month[:5])
        week_emb = self.week_emb(torch.tensor(week, device=self.device))
        day_emb = self.day_emb(torch.tensor(day - 1, device=self.device))
        month_emb = self.month_emb(torch.tensor(month - 1, device=self.device))
        return month_emb + week_emb + day_emb


class TrendEncoder(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension, device, step_length, max_length):
        super(TrendEncoder, self).__init__()
        self.max_length = max_length
        self.step_length = step_length
        assert type(max_length) == int and type(step_length) == int and max_length % step_length == 0
        self.dimension = dimension
        self.encoder = nn.LSTM(input_size=1, hidden_size=dimension, batch_first=True)
        self.device = device

    def forward(self, time, length, predict_time):
        # time has shape [batch_size,step]
        time = time.cpu().numpy()
        predict_time = predict_time.cpu().numpy()
        batch, _ = time.shape
        time_emb = np.zeros((batch, self.max_length // self.step_length))
        for i in range(batch):
            for j in range(length[i]):
                pos = int((predict_time[i] - time[i][j]) / self.step_length)
                if pos < time_emb.shape[1]:
                    time_emb[i][-(pos + 1)] += 1
        # [batch,L]
        time_emb = torch.tensor(time_emb, dtype=torch.float, device=self.device)
        _, (h, _) = self.encoder.forward(time_emb[:, :, None])
        return h.squeeze(dim=0)



def get_time_encoder(model_type, dimension, max_time=None, time_num=20, single=False):
    if model_type == 'tgat':
        user_time_encoder = TGATEncoder(dimension)
        if single:
            cas_time_encoder = user_time_encoder
        else:
            cas_time_encoder = TGATEncoder(dimension)
        return nn.ModuleDict({'user': user_time_encoder, 'cas': cas_time_encoder})
    elif model_type == 'emb':
        return nn.ModuleDict({
            'user': EmbedEncoder(dimension, max_time['user'], time_num),
            'cas': EmbedEncoder(dimension, max_time['cas'], time_num)
        })
    else:
        raise ValueError("Not Implemented Model Type")
