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


class Fourier3dCustom(torch.nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.split = kwargs.get("split", False)

	def forward(self, inputs):
		x = inputs
		x = self.fourier_split(x) if self.split else self.fourier_merge(x)
		return x

	def fourier_split(self, inputs):
		ffted = torch.fft.fftn(inputs)
		return ffted.real, ffted.imag

	def fourier_merge(self, inputs):
		real, imag = inputs[0], inputs[1]
		precomp = torch.stack((real, imag), dim=-1)
		complx = torch.view_as_complex(precomp)
		iffted = torch.fft.ifftn(complx, dim=(-3, -2, -1))
		return iffted


class RightModel(torch.nn.Module):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.fsplit = Fourier3dCustom(**dict(split=True))
		self.downsample = nn.AdaptiveAvgPool3d(**dict(output_size=64))
		self.conv_block = nn.Sequential(
			Conv3dCustom(**dict(in_feat=1, out_feat=128, kernel=7, stride=1, batch_norm=False, transpose=False)),
			Conv3dCustom(**dict(in_feat=128, out_feat=1, kernel=7, stride=1, batch_norm=False, transpose=True)),
			Conv3dCustom(**dict(in_feat=1, out_feat=128, kernel=7, stride=1, batch_norm=False, transpose=False)),
			Conv3dCustom(**dict(in_feat=128, out_feat=1, kernel=7, stride=1, batch_norm=False, transpose=True)),
		)
		self.dropout = nn.Dropout(opt.dropout)
		self.upsample = torch.nn.Upsample(**dict(size=self.opt.pad_size, mode='trilinear', align_corners=False))
		self.fmerge = Fourier3dCustom(**dict(split=False))
		self.bn = torch.nn.BatchNorm3d(**dict(num_features=1))

	def forward(self, inputs):
		r, i = self.fsplit(inputs)
		r, i = list(map(lambda inp: self.downsample(inp), (r, i)))
		r, i = list(map(lambda inp: self.conv_block(inp), (r, i)))
		r, i = list(map(lambda inp: self.dropout(inp), (r, i)))
		r, i = list(map(lambda inp: self.upsample(inp), (r, i)))
		x = self.fmerge([r, i]).real
		x = self.bn(x)

		return x


class JointModel(torch.nn.Module):
	def __init__(self, opt):
		super().__init__()
		self.left_model = LeftModel(opt)
		self.right_model = RightModel(opt)

	def forward(self, inputs):
		out1 = self.left_model(inputs)
		out2 = self.right_model(inputs)

		pad, is_odd = divmod((out2.size(-1)-out1.size(-1)), 2)
		pad1, pad2 = pad, -pad-1 if is_odd else -pad
		out2 = out2[..., pad1:pad2, pad1:pad2, pad1:pad2]

		return out1 + out2


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

	model = JointModel(opt)
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
