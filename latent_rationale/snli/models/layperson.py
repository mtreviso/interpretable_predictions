#!/usr/bin/env python

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class LinearLayperson(nn.Module):
    """
    Simple linear classifier (uses a LSTM to encode hypothesis)
    """

    def __init__(self, cfg, vocab):
        super(LinearLayperson, self).__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.n_embed, cfg.embed_size,
                                  padding_idx=cfg.pad_idx)
        self.vocab = vocab
        self.pad_idx = cfg.pad_idx
        self.embed.weight.requires_grad = False

        self.rnn_type = 'lstm'
        self.rnn_hyp = nn.LSTM(cfg.embed_size, cfg.hidden_size,
                               bidirectional=cfg.birnn, batch_first=True)
        n = 2 if cfg.birnn else 1

        self.linear_message = nn.Linear(cfg.n_embed, cfg.output_size)
        self.linear_hyp = nn.Linear(n * cfg.hidden_size, cfg.output_size)
        self.linear_merge = nn.Linear(cfg.output_size, cfg.output_size)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, message, batch):

        # prem_input, prem_lengths = batch.premise
        hypo_input, hypo_lengths = batch.hypothesis

        # prem_mask = (prem_input != self.pad_idx)
        # hypo_mask = (hypo_input != self.pad_idx)

        # prem_embed = self.embed(prem_input)
        hypo_embed = self.embed(hypo_input)

        # do not backpropagate through embeddings when fixed
        # prem_embed = prem_embed.detach()
        hypo_embed = hypo_embed.detach()

        # premise = self.encoder(prem_embed, prem_lengths, prem_mask)
        # hypothesis = self.encoder(hypo_embed, hypo_lengths, hypo_mask)
        hypothesis = pack(hypo_embed, hypo_lengths, batch_first=True, enforce_sorted=False)
        hypothesis, hidden_hypo = self.rnn_hyp(hypothesis)
        hypothesis, _ = unpack(hypothesis, batch_first=True)
        if self.rnn_type == 'lstm':
            hidden_hypo = hidden_hypo[0]
        if self.rnn_hyp.bidirectional:
            hidden_hypo = [hidden_hypo[0], hidden_hypo[1]]
        else:
            hidden_hypo = [hidden_hypo[0]]
        hidden_hypo = torch.cat(hidden_hypo, dim=-1)

        message_logits = self.linear_message(message)
        hyp_logits = self.linear_hyp(hidden_hypo)
        merged_logits = torch.tanh(message_logits + hyp_logits)
        logits = self.linear_merge(merged_logits)

        return logits

    def get_loss(self, logits, targets):
        optional = {}
        batch_size = logits.size(0)
        loss = self.criterion(logits, targets) / batch_size
        optional["ce"] = loss.item()

        return loss, optional
