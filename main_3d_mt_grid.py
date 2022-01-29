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
from main_3d_mt import RegDataset, single_train, single_val, single_test
from unet import UNet


if __name__ == '__main__':
    for weight_decay in [0, 2e-5, 5e-5, 5e-4]:
        for loss05 in np.arange(0, 2.5, 0.5):
            for loss_peak in np.arange(0, 0.05, 0.01):
                os.system('CUDA_VISIBLE_DEVICES=0 python main_3d_mt.py --train_frac 0.1 --batch_size 28 --pad_size 64 --gpu --aug --unet_atten --epoches 20 --warm_up_split 5 --lr 1e-3 --weight_decay {} --loss05 {} --loss_peak {}'.format(weight_decay, loss05, loss_peak))
