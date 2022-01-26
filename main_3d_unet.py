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

from main_3d import get_parser, data_split, RegDataset, LeftModel, single_train, single_val, single_test, Conv3dCustom, get_loss_func
from unet import UNet


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

    model = UNet(opt, attention=opt.unet_atten)
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