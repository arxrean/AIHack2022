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

from main_3d import get_parser, data_split, LeftModel, Conv3dCustom, get_loss_func, unpad_input
from unet import UNet


# h5 later
class RegDataset(torch.utils.data.Dataset):
    def __init__(self, opt, data, mode='train'):
        self.opt = opt
        self.mode = mode
        self.data = data

        self.h5s = [h5py.File(opt.h5_path, "r")["time{}".format(i)] for i in range(10)]

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
        time, time_idx = divmod(self.data[idx], self.opt.h5_size)
        prev = self.h5s[time+int(self.opt.h5_times[0])-2][time_idx]
        curr = self.h5s[time+int(self.opt.h5_times[0])-1][time_idx]
        label = self.h5s[time+int(self.opt.h5_times[0])][time_idx]
        curr = np.stack((prev, curr))

        if self.opt.data_norm:
            curr = (curr-self.mean)/self.std
            label = (label-self.mean)/self.std

        img_size, label_size = curr.shape[-1], label.shape[-1]
        assert img_size == label_size
        if self.opt.aug and self.mode == 'train':
            dims = sorted(np.random.choice(3, np.random.randint(0, 4), replace=False))
            _dims = [x+1 for x in dims]
            if len(dims) > 0:
                curr = np.flip(curr, _dims).copy()
                label = np.flip(label, dims).copy()

        label = np.expand_dims(label, axis=0)
        return curr, label, img_size, time


def single_train(opt, model, loader, loss_func, optimizer, scheduler):
    model = model.train()
    for step, pack in enumerate(loader):
        images, labels, crop_sizes, times = pack
        if opt.gpu:
            images, labels = images.float().cuda(), labels.float().cuda()
        outs, labels = unpad_input(opt, model(
            images), labels, crop_sizes, 'train')
        # get_structure_factor(labels.cpu().numpy()[0, 0])
        loss_mse, loss05, loss_peak = loss_func(opt, outs, labels)
        if opt.log:
            wandb.log({'train_loss_mse': loss_mse.item()})
        loss = loss_mse
        if opt.loss05 != 0:
            loss += loss05
        if opt.loss_peak != 0:
            loss += loss_peak
        if opt.time_weight:
            loss *= (1+times.float().mean()/10)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if opt.log:
            wandb.log({'train_loss': loss.item()})
            if opt.loss05: 
                wandb.log({'train_loss05': loss05.item()})
            if opt.loss_peak:
                wandb.log({'train_loss_peak': loss_peak.item()})

    return model


def single_val(opt, model, loader, loss_func, optimizer, scheduler):
    model = model.eval()
    _loss = 0
    with torch.no_grad():
        for step, pack in enumerate(loader):
            images, labels, crop_sizes, times = pack
            if opt.gpu:
                images, labels = images.float().cuda(), labels.float().cuda()
            outs, labels = unpad_input(opt, model(
                images), labels, crop_sizes, 'val')
            loss_mse, loss05, loss_peak = loss_func(opt, outs, labels)
            if opt.log:
                wandb.log({'train_loss_mse': loss_mse.item()})
            loss = loss_mse
            if opt.loss05 != 0:
                loss += loss05
            if opt.loss_peak != 0:
                loss += loss_peak
            if opt.time_weight:
                loss *= (1+times.float().mean()/10)

            _loss += loss.item()
            if opt.log:
                wandb.log({'val_loss': loss.item()})
                if opt.loss05: 
                    wandb.log({'val_loss05': loss05.item()})
                if opt.loss_peak:
                    wandb.log({'val_loss_peak': loss_peak.item()})

    return _loss/len(loader)


def single_test(opt, model, loader, loss_func, optimizer, scheduler):
    model = model.eval()
    inputs, preds, gts = [], [], []
    with torch.no_grad():
        for step, pack in enumerate(loader):
            images, labels, crop_sizes, times = pack
            if opt.gpu:
                images, labels = images.float().cuda(), labels.float().cuda()
            outs, labels = unpad_input(opt, model(
                images), labels, crop_sizes, 'test')
            inputs.append(images.cpu().numpy())
            preds.append(outs.cpu().numpy())
            gts.append(labels.cpu().numpy())
    if opt.data_norm:
        inputs = (np.concatenate(inputs, 0)
                 * test_set.std) + test_set.mean
        preds = (np.concatenate(preds, 0)
                 * test_set.std) + test_set.mean
        gts = (np.concatenate(gts, 0)
               * test_set.std) + test_set.mean
    else:
        inputs = np.concatenate(inputs, 0)
        preds = np.concatenate(preds, 0)
        gts = np.concatenate(gts, 0)

    mse = mean_squared_error(gts.reshape(-1), preds.reshape(-1))
    rmse = mean_squared_error(
        gts.reshape(-1), preds.reshape(-1), squared=False)
    mae = mean_absolute_error(gts.reshape(-1), preds.reshape(-1))
    r2 = r2_score(gts.reshape(-1), preds.reshape(-1))

    print('MSE: {} RMSE: {} MAE: {} R2: {}'.format(mse, rmse, mae, r2))

    if opt.log:
        wandb.config.update({'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2})

    if opt.log and opt.plot:
        inputs = inputs[:, 0]
        preds = preds[:, 0]
        gts = gts[:, 0]
        plots = np.random.choice(range(len(inputs)), opt.plot_num)
        plt = plot_pairs(opt, inputs[plots], preds[plots], gts[plots])


def tile(data, size=2):

    return data.repeat(1, *([size]*3))


if __name__ == '__main__':
    if not os.path.exists('./dataset/data.h5'):
        print('get the data.h5 file and put it on ./dataset/data.h5')
        raise

    opt = get_parser()
    if opt.log:
        wandb.init(project="AIHack", name=opt.name)
        wandb.config.update(opt)

    data = np.arange(opt.h5_size*(int(opt.h5_times[1])-int(opt.h5_times[0])+1))

    _, _, test = data_split(opt, data)

    test_set = RegDataset(opt, test, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    model = UNet(opt, attention=opt.unet_atten, in_channels=opt.cin)
    if opt.gpu:
        model = model.cuda()

    # qrfzmwbgpd(alldata)/zqbsjtcwbe(stepbystep)
    best_model_param = torch.load('./save/models/zqbsjtcwbe.pth')
    model.load_state_dict(best_model_param)
    model = model.eval()
    single_test(opt, model, test_loader, None, None, None)

    preds = []
    with torch.no_grad():
        for step, pack in enumerate(test_loader):
            if opt.test_samples > 0 and step == opt.test_samples:
                break
            pred = []
            outs, labels, crop_sizes, times = pack
            if opt.tile >= 2:
                outs = tile(outs[0], opt.tile).unsqueeze(0)
            if opt.gpu:
                outs = outs.float().cuda()
            for _ in range(opt.test_steps):
                _outs = model(outs)
                pred.append(_outs.squeeze(1).cpu().numpy())
                outs = torch.stack((outs[0][1], _outs[0][0])).unsqueeze(0)
            pred = np.concatenate(pred, 0)
            preds.append(pred)
    preds = np.stack(preds).transpose((1, 0, 2, 3, 4))
    if not os.path.exists('./save/results/'):
        os,makedirs('./save/results/')
    filename = './save/results/zqbsjtcwbe'

    if opt.tile >= 2:
        filename += '_tile'
    filename += '.h5'
    hf = h5py.File(filename, 'w')
    for i in range(len(preds)):
        hf.create_dataset('time{}'.format(i), data=preds[i])
    hf.close()