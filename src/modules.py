import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_hidden, max_len=20000):
        super().__init__()

        pe = torch.zeros(max_len, d_hidden)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hidden, 2).float() * (-math.log(10000.0) / d_hidden))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1], :]

class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    """
    def __init__(self, n_vocab, d_emb, d_hidden, n_conv_layers=3, kernel=5, dropout=0.2):
        super(EncoderPrenet, self).__init__()
        self.emb = nn.Embedding(n_vocab, d_emb)
        
        layers = []
        in_c = [d_emb] + [d_hidden]*(n_conv_layers - 1)
        out_c = [d_hidden]*n_conv_layers
        for i,o in zip(in_c, out_c):
            layers += [
                nn.Conv1d(in_channels=i,
                          out_channels=o,
                          kernel_size=kernel,
                          padding=kernel // 2
                ),
                nn.BatchNorm1d(d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.layers = nn.Sequential(*layers)
        self.projection = nn.Linear(d_hidden, d_hidden)

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        x = self.layers(x)
        return self.projection(x.transpose(1, 2)) 


class DecoderPrenet(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
             nn.Linear(d_input, d_hidden),
             nn.ReLU(),
             nn.Dropout(dropout),
             nn.Linear(d_hidden, d_output),
             nn.ReLU(),
             nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)

class Slice(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
    def forward(self, x):
        return x[:, : , :-self.a]


class DecoderPostNet(nn.Module):
    def __init__(self, n_mel, d_hidden, kernel=5, n_conv_layers=3, dropout=0.1):
        super().__init__()

        layers = []
        in_c = [n_mel] + [d_hidden]*(n_conv_layers-2)
        out_c = [d_hidden]*(n_conv_layers-1)
        for i,o in zip(in_c, out_c):
            layers += [
                nn.Conv1d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=kernel,
                    padding=kernel-1 ## causal
                ), 
                Slice(kernel-1),
                nn.BatchNorm1d(d_hidden),
                nn.Tanh(),
                nn.Dropout(dropout)
            ]
        layers += [
            nn.Conv1d(
                in_channels=d_hidden,
                out_channels=n_mel,
                kernel_size=kernel,
                padding=kernel-1
            ), 
            Slice(kernel-1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
