#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.nn import Module, Parameter
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, L2=0.0):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.stdv = 1.0 / math.sqrt(self.hidden_size)
        self.L2 = L2

        self.mask = None  # will be set during forward
        self.alias = None  # will be set during forward
        self.item = None  # will be set during forward
        self.tar = None  # will be set during forward

    def forward(self, re_embedding, tar, train=True):
        rm = torch.sum(self.mask, dim=1)

        # Adjust for zero-based indexing
        last_id = self.alias[torch.arange(self.batch_size), (rm - 1).long()]
        last_h = re_embedding[torch.arange(self.batch_size), last_id]
        last_h = last_h.view(-1, self.out_size)

        b = self.weights['embedding'][1:]  # Assuming weights is defined elsewhere
        logits = torch.matmul(last_h, b.t())

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, tar - 1)

        # Add L2 Regularization if necessary
        if train:
            lossL2 = sum(torch.norm(v) for name, v in self.named_parameters() if not any(
                exclude_name in name for exclude_name in ['bias', 'gamma', 'b', 'g', 'beta'])) * self.L2
            loss = loss + lossL2

        return loss, logits


class SGNREC(Model):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, n_node=None,
                 lr=0.001, l2=0.0, layers=1, decay=None, lr_dc=0.1):
        super(SGNREC, self).__init__(hidden_size, out_size, batch_size, L2=l2)

        self.adj_in = None  # These tensors will be set during training/evaluation
        self.adj_out = None

        self.n_node = n_node
        self.layers = layers
        self.weights = self._init_weights()

        # Optimizer setup
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _init_weights(self):
        all_weights = {}
        initializer = torch.FloatTensor(self.n_node, self.hidden_size).uniform_(-self.stdv, self.stdv)

        all_weights['embedding'] = nn.Parameter(initializer)
        all_weights['W_1'] = nn.Parameter(
            torch.FloatTensor(2 * self.out_size, 2 * self.out_size).uniform_(-self.stdv, self.stdv))
        all_weights['W_2'] = nn.Parameter(
            torch.FloatTensor(2 * self.out_size, self.out_size).uniform_(-self.stdv, self.stdv))

        return all_weights

    def sgc(self):
        fin_state = self.weights['embedding'][self.item]  # Assuming self.item is already tensor with required indices
        fin_state = fin_state.view(self.batch_size, -1, self.out_size)
        fin_state_in = fin_state
        fin_state_out = fin_state

        adj_in = self.adj_in ** self.layers
        adj_out = self.adj_out ** self.layers

        fin_state_in = torch.matmul(adj_in, fin_state_in)
        fin_state_out = torch.matmul(adj_out, fin_state_out)
        av = torch.cat([fin_state_in, fin_state_out], dim=-1)
        av = nn.functional.relu(torch.matmul(av, self.weights['W_1']))
        av = torch.matmul(av, self.weights['W_2'])

        return av.view(self.batch_size, -1, self.out_size)

    def run(self, tar, item, adj_in, adj_out, alias, mask):
        # Set these variables before calling forward
        self.tar = tar
        self.item = item
        self.adj_in = adj_in
        self.adj_out = adj_out
        self.alias = alias
        self.mask = mask

        # Forward pass
        re_embedding = self.sgc()
        loss, logits = self.forward(re_embedding, tar)

        return loss, logits


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.optimizer.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
