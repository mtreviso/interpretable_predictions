#!/usr/bin/env python

import torch
from torch import nn


class LinearLayperson(nn.Module):
    """
    Simple linear classifier
    """

    def __init__(self, vocab, vocab_size, output_size):
        super(LinearLayperson, self).__init__()
        self.vocab = vocab
        self.net = nn.Sequential(
            nn.Linear(vocab_size, output_size),
            nn.LogSoftmax(dim=-1)
        )
        self.criterion = nn.NLLLoss(reduction='none')

    def forward(self, x, **kwargs):
        """
        :param x: [B, VOCAB_SIZE] (bag of words)
        :return:
        """
        y = self.net(x)
        return y

    def predict(self, logits):
        """
        Predict.
        :param logits:
        :return:
        """
        return logits.argmax(dim=-1)

    def get_loss(self, logits, targets, **kwargs):

        optional = dict()
        loss = self.criterion(logits, targets)

        return loss.mean(), optional
