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

# python data_stat.py --num_workers 0 --epoches 2 --pad_size 64 --batch_size 32
if __name__ == '__main__':
    opt = get_parser()
    for t in range(1, 10):
        opt.h5_time = t
        data = np.arange(opt.h5_size)

        train, val, test = data_split(opt, data)

        train_set = RegDataset(opt, train, 'train')
        train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

        prev, curr = [], []
        for step, pack in enumerate(train_loader):
            images, labels, crop_sizes = pack
            prev.append(images.numpy())
            curr.append(labels.numpy())
        prev = np.concatenate(prev, 0)
        curr = np.concatenate(curr, 0)

        print('------time: {}------'.format(opt.h5_time-1))
        print('min: {} max: {} mean: {} median: {}'.format(np.min(prev), np.max(prev), np.mean(prev), np.median(prev)))
        hist, bins = np.histogram(prev.reshape(-1), bins = np.arange(0, 2.7, 0.1))
        print([(str(h), '{:.1f}'.format(b)) for h, b in zip(hist, bins)])

'''
------time: 0------
min: 0.0 max: 2.496000051498413 mean: 0.4999995529651642 median: 0.4480000138282776
[('2442114', '0.0'), ('15985393', '0.1'), ('16180017', '0.2'), ('45264572', '0.3'), ('24964575', '0.4'), ('44372539', '0.5'), ('17102776', '0.6'), ('23177130', '0.7'), ('11734566', '0.8'), ('3118305', '0.9'), ('3187389', '1.0'), ('722099', '1.1'), ('657959', '1.2'), ('132280', '1.3'), ('109380', '1.4'), ('19827', '1.5'), ('15239', '1.6'), ('3740', '1.7'), ('542', '1.8'), ('373', '1.9'), ('52', '2.0'), ('34', '2.1'), ('7', '2.2'), ('3', '2.3'), ('1', '2.4'), ('0', '2.5')]
------time: 1------
min: 0.0 max: 2.559999942779541 mean: 0.49999991059303284 median: 0.4480000138282776
[('51215477', '0.0'), ('26902770', '0.1'), ('9409330', '0.2'), ('15717476', '0.3'), ('7270183', '0.4'), ('14830385', '0.5'), ('7927298', '0.6'), ('17089585', '0.7'), ('17671401', '0.8'), ('8325199', '0.9'), ('14063889', '1.0'), ('5370007', '1.1'), ('7340687', '1.2'), ('2224985', '1.3'), ('2475462', '1.4'), ('597974', '1.5'), ('547109', '1.6'), ('160832', '1.7'), ('27245', '1.8'), ('19166', '1.9'), ('2609', '2.0'), ('1546', '2.1'), ('192', '2.2'), ('96', '2.3'), ('6', '2.4'), ('3', '2.5')]
------time: 2------
min: 0.0 max: 2.687999963760376 mean: 0.5000014901161194 median: 0.4480000138282776
[('51630789', '0.0'), ('26758495', '0.1'), ('9319556', '0.2'), ('15567781', '0.3'), ('7203683', '0.4'), ('14721106', '0.5'), ('7884472', '0.6'), ('17064118', '0.7'), ('17710369', '0.8'), ('8349680', '0.9'), ('14131767', '1.0'), ('5401628', '1.1'), ('7372093', '1.2'), ('2232946', '1.3'), ('2482262', '1.4'), ('598646', '1.5'), ('549986', '1.6'), ('161048', '1.7'), ('27208', '1.8'), ('18913', '1.9'), ('2561', '2.0'), ('1520', '2.1'), ('177', '2.2'), ('96', '2.3'), ('11', '2.4'), ('0', '2.5')]
------time: 3------
min: 0.0 max: 2.559999942779541 mean: 0.500000536441803 median: 0.4480000138282776
[('51873466', '0.0'), ('26693602', '0.1'), ('9259430', '0.2'), ('15464998', '0.3'), ('7163983', '0.4'), ('14653152', '0.5'), ('7869135', '0.6'), ('17046392', '0.7'), ('17734019', '0.8'), ('8368119', '0.9'), ('14168253', '1.0'), ('5411635', '1.1'), ('7396228', '1.2'), ('2242559', '1.3'), ('2487642', '1.4'), ('598398', '1.5'), ('549197', '1.6'), ('160367', '1.7'), ('27113', '1.8'), ('18756', '1.9'), ('2602', '2.0'), ('1581', '2.1'), ('188', '2.2'), ('87', '2.3'), ('9', '2.4'), ('1', '2.5')]
------time: 4------
min: 0.0 max: 2.496000051498413 mean: 0.5000011920928955 median: 0.4480000138282776
[('52080216', '0.0'), ('26643964', '0.1'), ('9214789', '0.2'), ('15374575', '0.3'), ('7117757', '0.4'), ('14595597', '0.5'), ('7851950', '0.6'), ('17041428', '0.7'), ('17748218', '0.8'), ('8383608', '0.9'), ('14202675', '1.0'), ('5422851', '1.1'), ('7413264', '1.2'), ('2245564', '1.3'), ('2492476', '1.4'), ('599846', '1.5'), ('550951', '1.6'), ('160667', '1.7'), ('27246', '1.8'), ('18840', '1.9'), ('2599', '2.0'), ('1562', '2.1'), ('180', '2.2'), ('83', '2.3'), ('6', '2.4'), ('0', '2.5')]
------time: 5------
min: 0.0 max: 2.559999942779541 mean: 0.4999995827674866 median: 0.4480000138282776
[('52285368', '0.0'), ('26586892', '0.1'), ('9166664', '0.2'), ('15292646', '0.3'), ('7084162', '0.4'), ('14531953', '0.5'), ('7829160', '0.6'), ('17034085', '0.7'), ('17770603', '0.8'), ('8400242', '0.9'), ('14236629', '1.0'), ('5436683', '1.1'), ('7426461', '1.2'), ('2248901', '1.3'), ('2497626', '1.4'), ('600565', '1.5'), ('551235', '1.6'), ('160828', '1.7'), ('27102', '1.8'), ('18730', '1.9'), ('2606', '2.0'), ('1489', '2.1'), ('172', '2.2'), ('95', '2.3'), ('13', '2.4'), ('2', '2.5')]
------time: 6------
min: 0.0 max: 2.559999942779541 mean: 0.49999943375587463 median: 0.4480000138282776
[('52487354', '0.0'), ('26535611', '0.1'), ('9130294', '0.2'), ('15199312', '0.3'), ('7041516', '0.4'), ('14476162', '0.5'), ('7809825', '0.6'), ('17017717', '0.7'), ('17799235', '0.8'), ('8417500', '0.9'), ('14259159', '1.0'), ('5450283', '1.1'), ('7449559', '1.2'), ('2254990', '1.3'), ('2499922', '1.4'), ('601073', '1.5'), ('550724', '1.6'), ('160429', '1.7'), ('27175', '1.8'), ('18667', '1.9'), ('2573', '2.0'), ('1550', '2.1'), ('174', '2.2'), ('92', '2.3'), ('14', '2.4'), ('2', '2.5')]
------time: 7------
min: 0.0 max: 2.559999942779541 mean: 0.5000017285346985 median: 0.4480000138282776
[('52715487', '0.0'), ('26475273', '0.1'), ('9077419', '0.2'), ('15103251', '0.3'), ('6997287', '0.4'), ('14415789', '0.5'), ('7789229', '0.6'), ('17008820', '0.7'), ('17810606', '0.8'), ('8435654', '0.9'), ('14306726', '1.0'), ('5463757', '1.1'), ('7466018', '1.2'), ('2258266', '1.3'), ('2503850', '1.4'), ('602611', '1.5'), ('549103', '1.6'), ('161278', '1.7'), ('27374', '1.8'), ('18703', '1.9'), ('2566', '2.0'), ('1567', '2.1'), ('184', '2.2'), ('80', '2.3'), ('13', '2.4'), ('1', '2.5')]
------time: 8------
min: 0.0 max: 2.559999942779541 mean: 0.5000019669532776 median: 0.4480000138282776
[('52934044', '0.0'), ('26423478', '0.1'), ('9024538', '0.2'), ('15011026', '0.3'), ('6959415', '0.4'), ('14337750', '0.5'), ('7774944', '0.6'), ('17000774', '0.7'), ('17837914', '0.8'), ('8457577', '0.9'), ('14331579', '1.0'), ('5475107', '1.1'), ('7485230', '1.2'), ('2262632', '1.3'), ('2508871', '1.4'), ('602661', '1.5'), ('552268', '1.6'), ('161164', '1.7'), ('26970', '1.8'), ('18585', '1.9'), ('2540', '2.0'), ('1542', '2.1'), ('197', '2.2'), ('98', '2.3'), ('7', '2.4'), ('1', '2.5')]
'''