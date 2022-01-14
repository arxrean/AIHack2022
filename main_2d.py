import os
import pdb
import copy
import random
import argparse
import pandas as pd
import numpy as np
from scipy.special import softmax
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default='tmp')
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument('--gpu', action='store_true')
	parser.add_argument('--gpus', action='store_true')
	parser.add_argument('--dist', action='store_true')

	parser.add_argument('--txt_path', type=str,
						default='./dataset/simulation/2d/density2d.txt')

	# data
	parser.add_argument('--train_test_ratio', type=float, default=0.3)
	parser.add_argument('--train_val_ratio', type=float, default=0.3)
	parser.add_argument('--warm_up_split', type=int, default=5)
	parser.add_argument('--img_size', type=int, default=64)
	parser.add_argument('--pad_size', type=float, default=128)
	parser.add_argument('--crop_size', type=int, default=64)
	parser.add_argument('--batches', type=int, default=32)
	parser.add_argument('--batch_num', type=int, default=16)
	parser.add_argument('--batch_thd', type=float, default=0)

	# train
	parser.add_argument('--epoches', type=int, default=40)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--train_frac', type=float, default=1.0)
	parser.add_argument('--weight_decay', type=float, default=2e-5)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--val_interval', type=float, default=200)
	parser.add_argument('--encode', type=str, default='ours_mm')
	parser.add_argument('--middle', type=str, default='pass')
	parser.add_argument('--decode', type=str, default='pass')

	# model
	parser.add_argument('--backbone', type=str, default='resnet18')
	parser.add_argument('--pretrain', action='store_true')
	parser.add_argument('--kpop', type=int, default=5)
	parser.add_argument('--embed_dim', type=int, default=32)
	parser.add_argument('--hidden_dim', type=int, default=32)
	parser.add_argument('--word_embed_size', type=int, default=128)
	parser.add_argument('--gru_hidden_dim', type=int, default=128)

	opt = parser.parse_args()

	return opt


def data_split(opt, data):
	train, test = train_test_split(data, test_size=opt.train_test_ratio, shuffle=False)
	train, val = train_test_split(train, test_size=opt.train_val_ratio, shuffle=False)

	return train, val, test


def unpad_loss(opt, outs, labels, mode='2d'):
	pad2, is_odd = divmod(outs.size(-1)-opt.img_size, 2)
	_outs = outs[:, :, pad2:-pad2-1 if is_odd else -pad2, pad2:-pad2-1 if is_odd else -pad2]

	return loss_func(_outs, labels)


def single_train(opt, model, train_loader):
	model = model.train()
	for step, pack in enumerate(train_loader):
		images, labels = pack
		if opt.gpu:
			images, labels = images.float().cuda(), labels.float().cuda()
		outs = model(images)
		loss = unpad_loss(opt, outs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()


class RegDataset(torch.utils.data.Dataset):
	def __init__(self, opt, data, mode='train'):
		self.opt = opt
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		idx = random.randint(0, len(self.data)-2)

		curr = self.data[idx]
		label = self.data[idx+1]

		if self.opt.pad_size != self.opt.img_size:
			assert (self.opt.pad_size - self.opt.img_size) % 2 == 0
			pad2 = (self.opt.pad_size - self.opt.img_size) // 2
			curr = np.pad(curr, ((0, 0), (pad2, pad2), (pad2, pad2)), 'constant')

		return curr, label


class Conv2dCustom(torch.nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.kernel = kwargs.get("kernel", 3)
		self.stride = kwargs.get("stride", 1)
		self.in_feat = kwargs.get("in_feat", 3)
		self.out_feat = kwargs.get("out_feat", 3)
		self.batch_norm = kwargs.get("batch_norm", True)
		self.transpose = kwargs.get("transpose", False)
		conv = torch.nn.Conv2d(self.in_feat, self.out_feat, self.kernel, stride=self.stride) if not self.transpose else torch.nn.ConvTranspose2d(
			self.in_feat, self.out_feat, self.kernel, stride=self.stride)
		if self.batch_norm:
			seq = torch.nn.Sequential(*[
				conv,
				torch.nn.BatchNorm2d(self.out_feat),
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


class LeftModel(torch.nn.Module):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.block1 = nn.Sequential(
			Conv2dCustom(**dict(in_feat=1, out_feat=128, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
			Conv2dCustom(**dict(in_feat=128, out_feat=64, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
		)
		self.block2 = nn.Sequential(
			Conv2dCustom(**dict(in_feat=64, out_feat=128, kernel=7,
								stride=2, batch_norm=True, transpose=False)),
			Conv2dCustom(**dict(in_feat=128, out_feat=64, kernel=7,
								stride=2, batch_norm=True, transpose=False)),
		)
		self.block3 = nn.Sequential(
			Conv2dCustom(**dict(in_feat=64, out_feat=64, kernel=7,
								stride=2, batch_norm=True, transpose=True)),
			Conv2dCustom(**dict(in_feat=64, out_feat=32, kernel=7,
								stride=2, batch_norm=True, transpose=True)),
		)
		self.block4 = nn.Sequential(
			Conv2dCustom(**dict(in_feat=32, out_feat=128, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
			Conv2dCustom(**dict(in_feat=128, out_feat=64, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
		)
		self.block5 = nn.Sequential(
			Conv2dCustom(**dict(in_feat=64, out_feat=1, kernel=7, stride=1,
								batch_norm=True, transpose=False)),
			torch.nn.BatchNorm2d(1),
			nn.ConvTranspose2d(1, 1, kernel_size=7, stride=1),
			nn.Tanh()
		)

	def forward(self, inputs):
		x = self.block1(inputs)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		return x


if __name__ == '__main__':
	opt = get_parser()

	data = np.loadtxt(opt.txt_path).transpose().reshape(-1, 1, 64, 64)
	train, val, test = data_split(opt, data)
	train_set, val_set, test_set = RegDataset(opt, train, 'train'), RegDataset(
		opt, val, 'val'), RegDataset(opt, test, 'test')
	train_loader, val_loader, test_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True), \
		DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers), \
		DataLoader(test_set, batch_size=opt.batch_size,
				   shuffle=False, num_workers=opt.num_workers)

	model = LeftModel(opt)
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

	for epoch in range(opt.epoches):
		score, pred, gt, model = single_train(opt, model, train_loader)

	l_model = LeftModel(left_config=left_config)
	r_model = RightModel(right_config=right_config)

	dataset = Dataset(data)
	dataloader = torch.utils.data.DataLoader(
		dataset, batch_size=3, shuffle=True, collate_fn=None)

	print(l_model(next(iter(dataloader))[:, 0, ..., 0]).size())
	print(r_model(next(iter(dataloader))[:, 0, ..., 0]).size())
