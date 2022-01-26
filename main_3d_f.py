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

from main_3d import get_parser, data_split, RegDataset, LeftModel, single_train, single_val, single_test, get_loss_func
from unet import UNet


class Conv3dCustom(torch.nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.kernel = kwargs.get("kernel", 3)
		self.stride = kwargs.get("stride", 1)
		self.in_feat = kwargs.get("in_feat", 3)
		self.out_feat = kwargs.get("out_feat", 3)
		self.batch_norm = kwargs.get("batch_norm", True)
		self.transpose = kwargs.get("transpose", False)
		if not self.transpose:
			conv = torch.nn.Conv3d(self.in_feat, self.out_feat, self.kernel, stride=self.stride) 
		else: 
			conv = torch.nn.ConvTranspose3d(self.in_feat, self.out_feat, self.kernel, stride=self.stride)
		
		if self.batch_norm:
			seq = torch.nn.Sequential(*[
				conv,
				torch.nn.BatchNorm3d(self.out_feat),
				torch.nn.LeakyReLU(True)
			])
		else:
			seq = torch.nn.Sequential(*[
				conv,
				torch.nn.LeakyReLU(True)
			])
		self.add_module("conv_unit", seq)

	def forward(self, inputs):
		x = inputs
		return self.conv_unit(x)


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
		# r, i = list(map(lambda inp: self.upsample(inp), (r, i)))
		x = self.fmerge([r, i]).real
		x = self.bn(x)

		if self.opt.activation != 'empty':
			x = self.use_activation(x)
		return x
		
	def use_activation(self, x):
		if self.opt.activation == 'sigmoid':
			return torch.sigmoid(x) * self.opt.scope

		raise


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

	model = RightModel(opt)
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

		wandb.log({'epoch_loss': val_loss})

	model.load_state_dict(best_model_param)
	single_test(opt, model, test_loader, get_loss_func, optimizer, scheduler)