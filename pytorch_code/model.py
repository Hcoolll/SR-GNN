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


class SGNREC(Module):
    def __init__(self, hidden_size, layers=1):
        super(SGNREC, self).__init__()

        self.hidden_size = hidden_size
        self.layers = layers
        self.batch_size = hidden_size
        self.out_size = hidden_size
        self.weights = self._init_weights()

    def _init_weights(self):
        all_weights = {}
        initializer = torch.FloatTensor(self.n_node, self.hidden_size).uniform_(-self.stdv, self.stdv)

        all_weights['embedding'] = nn.Parameter(initializer)
        all_weights['W_1'] = nn.Parameter(
            torch.FloatTensor(2 * self.out_size, 2 * self.out_size).uniform_(-self.stdv, self.stdv))
        all_weights['W_2'] = nn.Parameter(
            torch.FloatTensor(2 * self.out_size, self.out_size).uniform_(-self.stdv, self.stdv))

        return all_weights

    def sgc(self, a_in, a_out, hidden):
        hidden = hidden.view(self.batch_size, -1, self.out_size)
        fin_state_in = hidden
        fin_state_out = hidden
        adj_in = a_in ** self.layers
        adj_out = a_out ** self.layers

        fin_state_in = torch.matmul(adj_in, fin_state_in)
        fin_state_out = torch.matmul(adj_out, fin_state_out)
        av = torch.cat([fin_state_in, fin_state_out], dim=-1)
        av = nn.functional.relu(torch.matmul(av, self.weights['W_1']))
        av = torch.matmul(av, self.weights['W_2'])

        return av.view(self.batch_size, -1, self.out_size)

    def forward(self, a_in, a_out, hidden):
        for i in range(self.layers):
            hidden = self.sgc(a_in, a_out, hidden)
        return hidden


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
        # self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

        self.sgn = SGNREC(hidden_size=opt.hiddenSize, layers=opt.layers)

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

    def forward(self, inputs, a_in, a_out):
        hidden = self.embedding(inputs)
        hidden = self.sgn(a_in, a_out, hidden)
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
    adj_in, adj_out, alias, item, mask, targets = data.get_slice(i)

    adj_in = trans_to_cuda(torch.Tensor(np.array(adj_in)).float())
    adj_out = trans_to_cuda(torch.Tensor(np.array(adj_out)).float())

    alias = trans_to_cuda(torch.Tensor(alias).long())
    item = trans_to_cuda(torch.Tensor(item).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    hidden = model(item, adj_in, adj_out)
    get = lambda i: hidden[i][alias[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias)).long()])
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
