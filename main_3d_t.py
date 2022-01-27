import os
import pdb
import glob
import copy
import wandb
import h5py
import random
import string
import argparse
import pandas as pd
import numpy as np
from scipy.special import softmax
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fast_ml.model_development import train_valid_test_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import scipy.stats
import scipy.signal
import scipy.optimize

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
from unet import UNet


# h5 later
class RegDataset(torch.utils.data.Dataset):
    def __init__(self, opt, data, mode='train'):
        self.opt = opt
        self.mode = mode
        self.data = data
        assert opt.h5_time >= 4 and opt.h5_time <= 9
        self.h5_prev_prev_prev = h5py.File(opt.h5_path, "r")["time{}".format(opt.h5_time-3)]
        self.h5_prev_prev = h5py.File(opt.h5_path, "r")["time{}".format(opt.h5_time-2)]
        self.h5_prev = h5py.File(opt.h5_path, "r")["time{}".format(opt.h5_time-1)]
        self.h5_curr = h5py.File(opt.h5_path, "r")["time{}".format(opt.h5_time)]

        if opt.data_norm:
            self.mean, self.std = self.data.reshape(
                -1).mean(), self.data.reshape(-1).std()

        if mode == 'train' and self.opt.train_frac < 1.0:
            self.data = self.data[np.random.choice(
                len(self.data), int(len(self.data)*self.opt.train_frac))]

        if self.opt.log:
            wandb.config.update({self.mode: len(self.data)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prev_prev = self.h5_prev_prev_prev[self.data[idx]]
        prev = self.h5_prev_prev[self.data[idx]]
        curr = self.h5_prev[self.data[idx]]
        label = self.h5_curr[self.data[idx]]
        curr = np.stack((prev_prev, prev, curr))

        if self.opt.data_norm:
            curr = (curr-self.mean)/self.std
            label = (label-self.mean)/self.std

        img_size, label_size = curr.shape[-1], label.shape[-1]
        assert img_size == label_size
        if self.opt.aug and self.mode == 'train':
            print('no aug for t')

        # curr = np.expand_dims(curr, axis=0)
        label = np.expand_dims(label, axis=0)
        return curr, label, img_size


if __name__ == '__main__':
    opt = get_parser()
    if opt.log:
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

    model = UNet(opt, attention=opt.unet_atten, in_channels=opt.cin)
    if opt.gpu:
        model = model.cuda()

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
        model = single_train(opt, model, train_loader, get_loss_func, optimizer, scheduler)
        val_loss = single_val(opt, model, val_loader, get_loss_func, optimizer, scheduler)
        if val_loss < best_loss:
            best_model_param = model.state_dict()
            best_loss = val_loss
        if opt.log:
            wandb.log({'epoch_loss': val_loss})

    model.load_state_dict(best_model_param)
    single_test(opt, model, test_loader, get_loss_func, optimizer, scheduler)