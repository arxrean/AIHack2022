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
from PIL import Image

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

from main_3d import get_parser, data_split, RegDataset, LeftModel, single_train, single_val, single_test, Conv3dCustom
from unet import UNet


if __name__ == '__main__':
    opt = get_parser()
    if opt.log:
        wandb.init(project="AIHack")
        wandb.config.update(opt)

    data = np.loadtxt(opt.txt_path).transpose().reshape(-1, 1, 64, 64, 64)
    data = np.array([data[i:i+2] for i in range(len(data)-1)])
    data2 = [np.loadtxt(x).transpose().reshape(-1, 1, 64, 64, 64)
             for x in glob.glob('./dataset/addtional_data/*/density3d.txt')]
    data2_pair = []
    for d in data2:
        for i in range(len(d)-1):
            data2_pair.append(d[i:i+2])
    data = np.concatenate((np.array(data2_pair), data), 0)

    train, val, test = data_split(opt, data)

    train_set, val_set, test_set = RegDataset(opt, train, 'train'), RegDataset(
        opt, val, 'val'), RegDataset(opt, test, 'test')
    val_set.mean, val_set.std = train_set.mean, train_set.std
    test_set.mean, test_set.std = train_set.mean, train_set.std
    train_loader, val_loader, test_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True), \
        DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True), \
        DataLoader(test_set, batch_size=opt.batch_size,
                   shuffle=False, num_workers=opt.num_workers)

    model = UNet(opt)
    if opt.gpu:
        model = model.cuda()

    loss_func = nn.MSELoss()

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
        model = single_train(opt, model, train_loader, loss_func, optimizer, scheduler)
        val_loss = single_val(opt, model, val_loader, loss_func, optimizer, scheduler)
        if val_loss < best_loss:
            best_model_param = model.state_dict()
            best_loss = val_loss

        wandb.log({'epoch_loss': val_loss})

    model.load_state_dict(best_model_param)
    single_test(opt, model, test_loader, loss_func, optimizer, scheduler)