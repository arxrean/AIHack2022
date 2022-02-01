#!/usr/bin/python
# coding=utf-8

from functools import reduce
#from utils import nice_print, mem_report, cpu_stats
import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

   
import os
import pdb
import glob
import copy
import wandb
import random
import argparse
import pandas as pd
import numpy as np
from scipy.special import softmax
#from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from main_3d import get_parser, data_split, RegDataset, LeftModel, single_train, single_val, single_test, Conv3dCustom, get_loss_func

class E3DLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, kernel_size, tau):
        super().__init__()

        self._tau = tau
        self._cells = []

        input_shape = list(input_shape)
        for i in range(num_layers):
            cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
            # NOTE hidden state becomes input to the next cell
            #input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input: "BTDDD"):
        # NOTE (batch, seq_len, input_shape)
        batch_size = input.size(0)
        c_history_states = []
        h_states = []
        outputs = []
        c_history, m, h = self._cells[0].init_hidden(
                    batch_size, self._tau, input.device
                )
        x = input
        for cell_idx, cell in enumerate(self._cells):
            c_history, m, h = cell(
                x, c_history, m, h
            )
            c_history_states.append(c_history)
            h_states.append(h)
        # NOTE Concat along the channels
        return torch.cat(h_states, dim=1)


class E3DLSTMCell(nn.Module):
    def __init__(self, input_shape, hidden_size, kernel_size):
        super().__init__()

        in_channels = input_shape[0]
        self._input_shape = input_shape
        self._hidden_size = hidden_size

        # memory gates: input, cell(input modulation), forget
        self.weight_xi = ConvDeconv3d(in_channels, hidden_size, kernel_size)
        self.weight_hi = ConvDeconv3d(hidden_size, hidden_size, kernel_size, bias=False)

        self.weight_xg = copy.deepcopy(self.weight_xi)
        self.weight_hg = copy.deepcopy(self.weight_hi)

        self.weight_xr = copy.deepcopy(self.weight_xi)
        self.weight_hr = copy.deepcopy(self.weight_hi)

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.weight_xi_prime = copy.deepcopy(self.weight_xi)
        self.weight_mi_prime = copy.deepcopy(self.weight_hi)

        self.weight_xg_prime = copy.deepcopy(self.weight_xi)
        self.weight_mg_prime = copy.deepcopy(self.weight_hi)

        self.weight_xf_prime = copy.deepcopy(self.weight_xi)
        self.weight_mf_prime = copy.deepcopy(self.weight_hi)

        self.weight_xo = copy.deepcopy(self.weight_xi)
        self.weight_ho = copy.deepcopy(self.weight_hi)
        self.weight_co = copy.deepcopy(self.weight_hi)
        self.weight_mo = copy.deepcopy(self.weight_hi)

        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_attention(self, r, c_history):
        batch_size = r.size(0)
        channels = r.size(1)
        r_flatten = r.view(batch_size, -1, channels)
        # BxtaoTHWxC
        c_history_flatten = c_history.view(batch_size, -1, channels)

        # Attention mechanism
        # BxTHWxC x BxtaoTHWxC' = B x THW x taoTHW
        scores = torch.einsum("bxc,byc->bxy", r_flatten, c_history_flatten)
        attention = F.softmax(scores, dim=2)

        return torch.einsum("bxy,byc->bxc", attention, c_history_flatten).view(*r.shape)

    def self_attention_fast(self, r, c_history):
        # Scaled Dot-Product but for tensors
        # instead of dot-product we do matrix contraction on twh dimensions
        scaling_factor = 1 / (reduce(operator.mul, r.shape[-3:], 1) ** 0.5)
        scores = torch.einsum("bctwh,lbctwh->bl", r, c_history) * scaling_factor

        attention = F.softmax(scores, dim=0)
        return torch.einsum("bl,lbctwh->bctwh", attention, c_history)

    def forward(self, x, c_history, m, h):
        # Normalized shape for LayerNorm is CxT×H×W
        normalized_shape = list(h.shape[-3:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)

        # R is CxT×H×W
        r = torch.sigmoid(LR(self.weight_xr(x) + self.weight_hr(h)))
        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))

        recall = self.self_attention_fast(r, c_history)

        c = i * g + self.layer_norm(c_history[-1] + recall)

        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(
            LR(
                self.weight_xo(x)
                + self.weight_ho(h)
                + self.weight_co(c)
                + self.weight_mo(m)
            )
        )
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)
        # nice_print(**locals())

        return (c_history, m, h)

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)

        return (c_history, m, h)


class ConvDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)
        # self.conv_transpose3d = nn.ConvTranspose3d(out_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")

class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return self.conv3d(input)
        # return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="trilinear")

input_shape = (3, 64, 64, 64)
tau = 2
hidden_size = 16
kernel = 5
lstm_layers = 2

#encoder = E3DLSTM(
#            input_shape, hidden_size, lstm_layers, kernel, tau
#        )
#decoder = Decoder3d(lstm_layers * hidden_size, 1, kernel_size=kernel, padding=((kernel-1)//2)) if (kernel-1)%2 == 0 else Exception() 

class Model(torch.nn.Module):
    def __init__(self, input_shape, hidden_size, lstm_layers, kernel, tau):
        super().__init__()
        self.add_module("encoder", E3DLSTM(input_shape, hidden_size, lstm_layers, kernel, tau))
        assert (kernel-1)%2 == 0, "Dimensionality..."
        self.add_module("decoder", Decoder3d(lstm_layers * hidden_size, 1, kernel_size=kernel, padding=((kernel-1)//2)))
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
        opt = get_parser()
        if opt.log:  
                wandb.login(key="f5582e0e72ff93ad157b67983b0c29b4dc40d4d7")
                wandb.init(project="AIHack", name=opt.name)
                wandb.config.update(opt)

        data = np.arange(opt.h5_size)

        train, val, test = data_split(opt, data)

        train_set, val_set, test_set = RegDataset(opt, train, 'train'), RegDataset(
                opt, val, 'val'), RegDataset(opt, test, 'test')
        if opt.data_norm:
                val_set.mean, val_set.std = train_set.mean, train_set.std
                test_set.mean, test_set.std = train_set.mean, train_set.std

        train_loader, val_loader, test_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True), \
                DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True), \
                DataLoader(test_set, batch_size=opt.batch_size,
                                   shuffle=False, num_workers=opt.num_workers)

        model = Model(input_shape, hidden_size, lstm_layers, kernel, tau)
        #print(next(iter(train_loader))[0].shape)
        if opt.gpu:
                model = model.cuda()

        #loss_func = nn.MSELoss()
        loss_func = get_loss_func

        optimizer = AdamW(model.parameters(), lr=opt.lr,
                                          weight_decay=opt.weight_decay)

        total_training_steps = len(train_loader) * opt.epoches
        warmup_steps = total_training_steps // opt.warm_up_split
        scheduler = get_linear_schedule_with_warmup(
               optimizer,
               num_warmup_steps=warmup_steps,
               num_training_steps=total_training_steps
        )

        best_model_param, best_loss = None, 1e5
        for epoch in range(opt.epoches):
                print(f"Epoch {epoch}...")
                model = single_train(opt, model, train_loader,
                                                         loss_func, optimizer, scheduler)
                val_loss = single_val(opt, model, val_loader,
                                                          loss_func, optimizer, scheduler)
                if val_loss < best_loss:
                        best_model_param = model.state_dict()
                        best_loss = val_loss
                if opt.log:
                        wandb.log({'epoch_loss': val_loss})

        model.load_state_dict(best_model_param)
        single_test(opt, model, test_loader, loss_func, optimizer, scheduler)





