# coding: utf-8
import math

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torch.nn import Linear, Sequential, ReLU, Dropout


class BernoulliAttention(nn.Module):
    """
    Computes Bernoulli Attention
    """

    def __init__(self, in_features, out_features, dropout=0.2):
        super(BernoulliAttention, self).__init__()

        self.activation = ReLU()
        self.dropout = Dropout(p=dropout)

        # self.a_layer = Sequential(
        #     Linear(in_features, out_features),
        #     self.activation, self.dropout,
        #     Linear(out_features, out_features),
        #     # self.activation, self.dropout
        # )

        self.q_layer = nn.Linear(in_features, out_features)
        self.k_layer = nn.Linear(in_features, out_features)
        # self.a_layer = nn.Linear(out_features, out_features)

        # self.W = nn.Parameter(torch.Tensor(out_features, out_features))
        # nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        self.dist = None

    def forward(self, q, k):
        # q_a = self.a_layer(q)
        # k_a = self.a_layer(k)
        # a = q_a @ k_a.transpose(1, 2)

        q_a = self.q_layer(q)
        k_a = self.k_layer(k)
        a = q_a @ k_a.transpose(1, 2)
        # a = torch.einsum('b...tm,nm,b...sn->b...ts', [q_a, self.W, k_a])
        # a = a / math.sqrt(k_a.shape[-1])

        # we return a distribution (from which we can sample if we want)
        dist = Bernoulli(logits=a)
        self.dist = dist

        if self.training:  # sample
            z = dist.sample()
        else:  # predict deterministically
            z = (dist.probs >= 0.5).float()

        return z
