import os
import pdb
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


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default='tmp')
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument('--gpu', action='store_true')
	parser.add_argument('--gpus', action='store_true')
	parser.add_argument('--log', action='store_true')
	parser.add_argument('--aug', action='store_true')

	parser.add_argument('--txt_path', type=str,
						default='./dataset/simulation/3d/density3d.txt')

	# data
	parser.add_argument('--train_test_ratio', type=float, default=0.3)
	parser.add_argument('--train_val_ratio', type=float, default=0.3)
	parser.add_argument('--warm_up_split', type=int, default=5)
	parser.add_argument('--img_size', type=int, default=64)
	parser.add_argument('--pad_size', type=float, default=128)
	parser.add_argument('--batches', type=int, default=32)
	parser.add_argument('--data_norm', action='store_true')

	# train
	parser.add_argument('--epoches', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--train_frac', type=float, default=1.0)
	parser.add_argument('--weight_decay', type=float, default=2e-5)
	parser.add_argument('--dropout', type=float, default=0.1)

	# model
	parser.add_argument('--backbone', type=str, default='resnet18')

	opt = parser.parse_args()

	return opt


def data_split(opt, data):
	train, test = train_test_split(
		data, test_size=opt.train_test_ratio, random_state=42)
	train, val = train_test_split(train, test_size=opt.train_val_ratio)

	return train, val, test


def unpad_input(opt, outs, labels, crop_sizes, mode='train'):
	if mode == 'train':
		outs_new, labels_new = [], []
		for out, label, crop_size in zip(outs, labels, crop_sizes):
			pad2, is_odd = divmod(out.size(-1)-crop_size.item(), 2)
			_out = out[:, pad2:-pad2-1 if is_odd else -
						 pad2, pad2:-pad2-1 if is_odd else -pad2, pad2:-pad2-1 if is_odd else -pad2]
			pad2, is_odd = divmod(labels.size(-1)-crop_size.item(), 2)
			_label = label[:, pad2:-pad2-1 if is_odd else -
						 pad2, pad2:-pad2-1 if is_odd else -pad2, pad2:-pad2-1 if is_odd else -pad2]
			outs_new.append(_out)
			labels_new.append(_label)
	else:
		pad2, is_odd = divmod(outs.size(-1)-crop_sizes[0].item(), 2)
		outs_new = outs[:, :, pad2:-pad2-1 if is_odd else -
						 pad2, pad2:-pad2-1 if is_odd else -pad2, pad2:-pad2-1 if is_odd else -pad2]
		pad2, is_odd = divmod(labels.size(-1)-crop_sizes[0].item(), 2)
		labels_new = labels[:, :, pad2:-pad2-1 if is_odd else -
						 pad2, pad2:-pad2-1 if is_odd else -pad2, pad2:-pad2-1 if is_odd else -pad2]

	return outs_new, labels_new


def single_train(opt, model, loader):
	model = model.train()
	for step, pack in enumerate(loader):
		images, labels, crop_sizes = pack
		if opt.gpu:
			images, labels = images.float().cuda(), labels.float().cuda()
		outs, labels = unpad_input(opt, model(images), labels, crop_sizes, 'train')
		loss = 0
		for out, label in zip(outs, labels):
			loss += loss_func(out, label)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		if opt.log:
			wandb.log({'train_loss': loss.item()})

	return model


def single_val(opt, model, loader):
	model = model.eval()
	_loss = 0
	with torch.no_grad():
		for step, pack in enumerate(loader):
			images, labels, crop_sizes = pack
			if opt.gpu:
				images, labels = images.float().cuda(), labels.float().cuda()
			outs, labels = unpad_input(opt, model(images), labels, crop_sizes, 'val')
			loss = loss_func(outs, labels).item()
			_loss += loss
			if opt.log:
				wandb.log({'val_loss': loss})

	return _loss/len(loader)


def single_test(opt, model, loader):
	model = model.eval()
	preds, gts = [], []
	with torch.no_grad():
		for step, pack in enumerate(loader):
			images, labels, crop_sizes = pack
			if opt.gpu:
				images, labels = images.float().cuda(), labels.float().cuda()
			outs, labels = unpad_input(opt, model(images), labels, crop_sizes, 'test')
			preds.append(outs.cpu().numpy())
			gts.append(labels.cpu().numpy())
	if opt.data_norm:
		preds = (np.concatenate(preds, 0).reshape(-1)
				 * test_set.std) + test_set.mean
		gts = (np.concatenate(gts, 0).reshape(-1)
			   * test_set.std) + test_set.mean
	else:
		preds = np.concatenate(preds, 0).reshape(-1)
		gts = np.concatenate(gts, 0).reshape(-1)

	print('MSE: {} RMSE: {} MAE: {} R2: {}'.format(mean_squared_error(gts, preds),
												   mean_squared_error(
		gts, preds, squared=False),
		mean_absolute_error(
		gts, preds),
		r2_score(gts, preds)))

	if opt.log:
		wandb.config.update({'mse': mean_squared_error(gts, preds),
							 'rmse': mean_squared_error(gts, preds, squared=False),
							 'mae': mean_absolute_error(gts, preds),
							 'r2': r2_score(gts, preds)})

# h5 later


class RegDataset(torch.utils.data.Dataset):
	def __init__(self, opt, data, mode='train'):
		self.opt = opt
		self.mode = mode
		self.data = data
		self.mean, self.std = self.data.reshape(
			-1).mean(), self.data.reshape(-1).std()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		curr = self.data[idx][0]
		label = self.data[idx][1]

		if self.opt.data_norm:
			curr = (curr-self.mean)/self.std
			label = (label-self.mean)/self.std

		img_size, label_size = curr.shape[-1], label.shape[-1]
		assert img_size == label_size

		if self.opt.aug and self.mode == 'train':
			crop_size = random.choice([16, 32, 64])
			if crop_size < img_size:
				top_left = [random.randint(0, img_size-crop_size) for _ in range(3)]
				curr = curr[:, top_left[0]:top_left[0]+crop_size, top_left[1]:top_left[1]+crop_size, top_left[2]:top_left[2]+crop_size]
				label = label[:, top_left[0]:top_left[0]+crop_size, top_left[1]:top_left[1]+crop_size, top_left[2]:top_left[2]+crop_size]
				img_size = crop_size

		if self.opt.pad_size != img_size:
			assert (self.opt.pad_size - img_size) % 2 == 0
			pad2 = (self.opt.pad_size - img_size) // 2
			curr = np.pad(curr, ((0, 0), (pad2, pad2), (pad2, pad2),
								 (pad2, pad2)), 'constant')
			label = np.pad(label, ((0, 0), (pad2, pad2), (pad2, pad2),
								 (pad2, pad2)), 'constant')

		return curr, label, img_size


class Conv3dCustom(torch.nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.kernel = kwargs.get("kernel", 3)
		self.stride = kwargs.get("stride", 1)
		self.in_feat = kwargs.get("in_feat", 3)
		self.out_feat = kwargs.get("out_feat", 3)
		self.batch_norm = kwargs.get("batch_norm", True)
		self.transpose = kwargs.get("transpose", False)
		conv = torch.nn.Conv3d(self.in_feat, self.out_feat, self.kernel, stride=self.stride) if not self.transpose else torch.nn.ConvTranspose3d(
			self.in_feat, self.out_feat, self.kernel, stride=self.stride)
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


class LeftModel(torch.nn.Module):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.block1 = nn.Sequential(
			Conv3dCustom(**dict(in_feat=1, out_feat=128, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
			Conv3dCustom(**dict(in_feat=128, out_feat=64, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
		)
		self.block2 = nn.Sequential(
			Conv3dCustom(**dict(in_feat=64, out_feat=128, kernel=7,
								stride=2, batch_norm=True, transpose=False)),
			Conv3dCustom(**dict(in_feat=128, out_feat=64, kernel=7,
								stride=2, batch_norm=True, transpose=False)),
		)
		self.block3 = nn.Sequential(
			Conv3dCustom(**dict(in_feat=64, out_feat=64, kernel=7,
								stride=2, batch_norm=True, transpose=True)),
			Conv3dCustom(**dict(in_feat=64, out_feat=32, kernel=7,
								stride=2, batch_norm=True, transpose=True)),
		)
		self.block4 = nn.Sequential(
			Conv3dCustom(**dict(in_feat=32, out_feat=128, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
			Conv3dCustom(**dict(in_feat=128, out_feat=64, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
		)
		self.block5 = nn.Sequential(
			Conv3dCustom(**dict(in_feat=64, out_feat=1, kernel=7, stride=1,
								batch_norm=True, transpose=False)),
			torch.nn.BatchNorm3d(1),
			nn.ConvTranspose3d(1, 1, kernel_size=7, stride=1),
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
	if opt.log:
		wandb.init(project="AIHack")
		wandb.config.update(opt)

	data = np.loadtxt(opt.txt_path).transpose().reshape(-1, 1, 64, 64, 64)
	data = np.array([data[i:i+2] for i in range(len(data)-1)])

	train, val, test = data_split(opt, data)

	train_set, val_set, test_set = RegDataset(opt, train, 'train'), RegDataset(
		opt, val, 'val'), RegDataset(opt, test, 'test')
	val_set.mean, val_set.std = train_set.mean, train_set.std
	test_set.mean, test_set.std = train_set.mean, train_set.std
	train_loader, val_loader, test_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True), \
		DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True), \
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

	best_model_param, best_loss = None, 1e5
	for epoch in range(opt.epoches):
		model = single_train(opt, model, train_loader)
		val_loss = single_val(opt, model, val_loader)
		if val_loss < best_loss:
			best_model_param = model.state_dict()
			best_loss = val_loss

		wandb.log({'epoch_loss': val_loss})

	model.load_state_dict(best_model_param)
	single_test(opt, model, test_loader)