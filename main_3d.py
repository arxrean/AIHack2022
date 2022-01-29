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


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default=''.join(random.choice(string.ascii_lowercase) for i in range(10)))
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument('--gpu', action='store_true')
	parser.add_argument('--gpus', action='store_true')
	parser.add_argument('--log', action='store_true')
	parser.add_argument('--aug', action='store_true')
	parser.add_argument('--plot', action='store_true')

	parser.add_argument('--txt_path', type=str,
						default='./dataset/simulation/3d/density3d.txt')
	parser.add_argument('--h5_path', type=str,
						default='./dataset/data.h5')
	parser.add_argument('--h5_size', type=int,
						default=1248)
	parser.add_argument('--h5_time', type=int,
						default=5)
	parser.add_argument('--h5_times', type=str, default='39')

	# data
	parser.add_argument('--train_test_ratio', type=float, default=0.02)
	parser.add_argument('--train_val_ratio', type=float, default=0.03)
	parser.add_argument('--train_frac', type=float, default=1.0)
	parser.add_argument('--warm_up_split', type=int, default=5)
	parser.add_argument('--img_size', type=int, default=64)
	parser.add_argument('--pad_size', type=float, default=128)
	parser.add_argument('--batches', type=int, default=32)
	parser.add_argument('--tile', type=int, default=2)
	parser.add_argument('--data_norm', action='store_true')

	# train
	parser.add_argument('--epoches', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--weight_decay', type=float, default=2e-5)
	parser.add_argument('--dropout', type=float, default=0)
	parser.add_argument('--plot_num', type=int, default=5)
	parser.add_argument('--loss05', type=float, default=0)
	parser.add_argument('--loss_peak', type=float, default=0)
	parser.add_argument('--loss_len', type=float, default=0)
	parser.add_argument('--time_weight', action='store_true')

	# model
	parser.add_argument('--backbone', type=str, default='resnet18')
	parser.add_argument('--activation', type=str, default='sigmoid')
	parser.add_argument('--scope', type=float, default=2.5)
	parser.add_argument('--unet_atten', action='store_true')
	parser.add_argument('--cin', type=int, default=2)

	opt = parser.parse_args()

	return opt


def get_loss_func(opt, outs, labels):
	loss_mse = nn.MSELoss()(outs, labels)
	loss05 = 0
	if opt.loss05 != 0:
		loss05 = opt.loss05*torch.abs(0.5-outs.reshape(outs.size(0), -1).mean(1)).mean()

	loss_peak = 0
	if opt.loss_peak != 0:
		preds = torch.stack([get_structure_factor_torch(opt, x[0])[1][1:] for x in outs])
		gts = torch.stack([get_structure_factor_torch(opt, x[0])[1][1:] for x in labels])
		loss_peak = opt.loss_peak * torch.abs(torch.argmax(preds, 1).float()-torch.argmax(gts, 1).float()).mean() 
		# loss += opt.loss_peak * torch.log(nn.MSELoss()(preds, gts)+1e-5)

	if opt.loss_len != 0:
		preds = torch.stack([get_correlation_length_torch(tuple(get_structure_factor_torch(opt, x[0]))) for x in outs])
		gts = torch.stack([get_correlation_length_torch(tuple(get_structure_factor_torch(opt, x[0]))) for x in labels])

	return loss_mse, loss05, loss_peak


def plot_pairs(opt, inputs, preds, gts):
	for i in range(len(inputs)):
		fig = plt.figure()
		ax = fig.add_subplot(1, 3, 1, projection='3d')
		X, Y, Z = np.meshgrid(
			*[np.linspace(0, 64, 64), np.linspace(0, 64, 64), np.linspace(0, 64, 64)])
		x, y, z = list(map(lambda inp: inp.reshape(-1, 1), (X, Y, Z)))
		g = inputs[i].reshape(-1, 1)
		ax.scatter(x, y, z, c=g, cmap='viridis', linewidth=0.5)

		ax = fig.add_subplot(1, 3, 2, projection='3d')
		X, Y, Z = np.meshgrid(
			*[np.linspace(0, 64, 64), np.linspace(0, 64, 64), np.linspace(0, 64, 64)])
		x, y, z = list(map(lambda inp: inp.reshape(-1, 1), (X, Y, Z)))
		g = preds[i].reshape(-1, 1)
		ax.scatter(x, y, z, c=g, cmap='viridis', linewidth=0.5)

		ax = fig.add_subplot(1, 3, 3, projection='3d')
		X, Y, Z = np.meshgrid(
			*[np.linspace(0, 64, 64), np.linspace(0, 64, 64), np.linspace(0, 64, 64)])
		x, y, z = list(map(lambda inp: inp.reshape(-1, 1), (X, Y, Z)))
		g = gts[i].reshape(-1, 1)
		ax.scatter(x, y, z, c=g, cmap='viridis', linewidth=0.5)

		folder = './save/{}/imgs'.format(opt.name)
		if not os.path.exists(folder):
			os.makedirs(folder)
		plt.savefig(os.path.join(folder, 'test_plots_{}.png'.format(i)), dpi=800)
		plt.close()


def data_split(opt, data):
	train, test = train_test_split(
		data, test_size=opt.train_test_ratio, random_state=42)
	train, val = train_test_split(train, test_size=opt.train_val_ratio, random_state=42)
	if opt.train_frac < 1:
		_, train = train_test_split(train, test_size=opt.train_frac, random_state=42)

	return train, val, test


def unpad_input(opt, outs, labels, crop_sizes, mode='train'):
	if tuple(outs.shape[-3:]) == tuple(labels.shape[-3:]):
		return outs, labels

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


def get_structure_factor(data, N=100):
	hat_data = np.fft.fftn(data)
	hat_data = np.fft.fftshift(hat_data)
	hat_data = np.abs(hat_data**2)
	# pdb.set_trace()
	center = np.asarray(data.shape)/2
	idx = np.indices(data.shape)
	q = np.sqrt(np.sum((idx - center[:, np.newaxis, np.newaxis, np.newaxis])**2, axis=0))
	max_q = np.max(q)

	q_array = np.linspace(0, max_q, N+1)
	sq_array = np.zeros(N)


	for i in range(len(sq_array)):
		sq_array[i] = np.mean(hat_data[np.logical_and(q_array[i] <= q, q <= q_array[i+1])])

	q_array = q_array[:-1] # + (q_array[1]-q_array[0])/2
	pdb.set_trace()

	# 64x64x64
	# label: 0.1, 0.1, 0.05, 0.7, 0.2, 0.1  3 
	# pred: 0.1, 0.8, 0.01, 0.1, 0.05, 0.08 1 -> 2
	return q_array, sq_array


def get_structure_factor_torch(opt, data, N=100):
	hat_data = torch.fft.fftn(data)
	hat_data = torch.fft.fftshift(hat_data)
	hat_data = torch.abs(hat_data**2)

	_data = data.detach().cpu().numpy()
	center = np.asarray(data.shape)/2
	idx = np.indices(_data.shape)
	q = np.sqrt(np.sum((idx - center[:, np.newaxis, np.newaxis, np.newaxis])**2, axis=0))
	max_q = np.max(q)

	q = torch.tensor(q)
	q_array = torch.tensor(np.linspace(0, max_q, N+1))
	if opt.gpu:
		q, q_array = q.cuda(), q_array.cuda()
	# sq_array = np.zeros(N)
	sq_array = torch.zeros(N).cuda() if opt.gpu else torch.zeros(N)

	for i in range(len(sq_array)):
		sq_array[i] = torch.mean(hat_data[torch.logical_and(q_array[i] <= q, q <= q_array[i+1])])

	q_array = q_array[:-1] # + (q_array[1]-q_array[0])/2
	return q_array, sq_array


def get_correlation_length(q, sq, plot_name=None):
	gr = np.fft.irfft(sq)
	gr = (gr[:gr.shape[0]//2] + np.flip(gr[gr.shape[0]//2:]))/2

	gr -= np.mean(gr)
	r = np.linspace(0, np.max(1/q[1:]), gr.shape[0])

	max_idx = scipy.signal.argrelextrema(gr, np.less)[0]
	popt, pcov = scipy.optimize.curve_fit(exp_decay, r[max_idx], gr[max_idx], p0=(-3e5, 10))

	if plot_name:
		fig, ax = plt.subplots()
		ax.set_xlabel("$r$ [arb. units]")
		ax.set_ylabel("$g(r)$ [arb. units]")

		ax.plot(r, gr, "o", label="data")
		ax.plot(r[max_idx], gr[max_idx], "o", label="max.")
		ax.plot(r, exp_decay(r, *popt), label="fit")
		ax.legend(loc="best")

		fig.savefig(plot_name, transparent=True, bbox_inches='tight', pad_inches=0)
		plt.close(fig)
	return popt[1]


def get_correlation_length_torch(inputs, plot_name=None):
	q, sq = inputs[0].cpu().numpy(), inputs[1]
	gr = torch.fft.irfft(sq)
	gr = (gr[:gr.shape[0]//2] + torch.flip(gr[gr.shape[0]//2:], dims=(0,)))/2

	gr = gr - gr.mean()
	r = np.linspace(0, np.max(1/q[1:]), gr.shape[0])

	max_idx = scipy.signal.argrelextrema(gr, torch.less)[0]
	popt, pcov = scipy.optimize.curve_fit(exp_decay, r[max_idx], gr[max_idx], p0=(-3e5, 10))

	if plot_name:
		fig, ax = plt.subplots()
		ax.set_xlabel("$r$ [arb. units]")
		ax.set_ylabel("$g(r)$ [arb. units]")

		ax.plot(r, gr, "o", label="data")
		ax.plot(r[max_idx], gr[max_idx], "o", label="max.")
		ax.plot(r, exp_decay(r, *popt), label="fit")
		ax.legend(loc="best")

		fig.savefig(plot_name, transparent=True, bbox_inches='tight', pad_inches=0)
		plt.close(fig)

	return popt[1]


def single_train(opt, model, loader, loss_func, optimizer, scheduler):
	model = model.train()
	for step, pack in enumerate(loader):
		images, labels, crop_sizes = pack
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
			images, labels, crop_sizes = pack
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
			images, labels, crop_sizes = pack
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


# h5 later
class RegDataset(torch.utils.data.Dataset):
	def __init__(self, opt, data, mode='train'):
		self.opt = opt
		self.mode = mode
		self.data = data
		assert opt.h5_time > 0 and opt.h5_time <= 9
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
		curr = self.h5_prev[self.data[idx]]
		label = self.h5_curr[self.data[idx]]

		if self.opt.data_norm:
			curr = (curr-self.mean)/self.std
			label = (label-self.mean)/self.std

		img_size, label_size = curr.shape[-1], label.shape[-1]
		assert img_size == label_size
		if self.opt.aug and self.mode == 'train':
			dims = tuple(sorted(np.random.choice(3, np.random.randint(0, 4), replace=False)))
			if len(dims) > 0:
				curr = np.flip(curr, dims).copy()
				label = np.flip(label, dims).copy()

		if self.opt.pad_size != img_size:
			assert (self.opt.pad_size - img_size) % 2 == 0
			pad2 = (self.opt.pad_size - img_size) // 2
			curr = np.pad(curr, ((0, 0), (pad2, pad2), (pad2, pad2),
								 (pad2, pad2)), 'constant')
			label = np.pad(label, ((0, 0), (pad2, pad2), (pad2, pad2),
								   (pad2, pad2)), 'constant')

		curr = np.expand_dims(curr, axis=0)
		label = np.expand_dims(label, axis=0)
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
		self.use_bias = kwargs.get("use_bias", False)
		self.padding = kwargs.get("padding", 0)
		self.output_padding = kwargs.get("output_padding", 0)
		if not self.transpose:
			conv = torch.nn.Conv3d(self.in_feat, self.out_feat, self.kernel, stride=self.stride, padding='same' if self.stride==1 else self.padding, bias=self.use_bias) 
		else: 
			conv = torch.nn.ConvTranspose3d(self.in_feat, self.out_feat, self.kernel, stride=self.stride, padding='same' if self.stride==1 else self.padding, bias=self.use_bias, output_padding=self.output_padding)
		
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
		pdb.set_trace()
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
								stride=2, batch_norm=True, transpose=False, padding=0)),
			Conv3dCustom(**dict(in_feat=128, out_feat=64, kernel=7,
								stride=2, batch_norm=True, transpose=False, padding=0)),
		)
		self.block3 = nn.Sequential(
			Conv3dCustom(**dict(in_feat=64, out_feat=64, kernel=7,
								stride=2, batch_norm=True, transpose=True, padding=0)),
			Conv3dCustom(**dict(in_feat=64, out_feat=32, kernel=7,
								stride=2, batch_norm=True, transpose=True, padding=0, output_padding=1)),
		)
		self.block4 = nn.Sequential(
			Conv3dCustom(**dict(in_feat=32, out_feat=128, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
			Conv3dCustom(**dict(in_feat=128, out_feat=64, kernel=7,
								stride=1, batch_norm=True, transpose=False)),
		)
		self.block5 = nn.Sequential(
			Conv3dCustom(**dict(in_feat=64, out_feat=1, kernel=1, stride=1,
								batch_norm=False, transpose=False)),
			torch.nn.BatchNorm3d(1),
			nn.ConvTranspose3d(1, 1, kernel_size=1, stride=1, bias=True),
			nn.Tanh()
		)

	def forward(self, inputs):
		x = self.block1(inputs) # torch.Size([4, 64, 64, 64, 64])
		x = self.block2(x) # torch.Size([4, 64, 12, 12, 12])
		x = self.block3(x) # torch.Size([4, 32, 63, 63, 63])
		x = self.block4(x)
		x = self.block5(x)
		return x


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
		model = single_train(opt, model, train_loader,
							 loss_func, optimizer, scheduler)
		val_loss = single_val(opt, model, val_loader,
							  loss_func, optimizer, scheduler)
		if val_loss < best_loss:
			best_model_param = model.state_dict()
			best_loss = val_loss

		if opt.log:
			wandb.log({'epoch_loss': val_loss})

	model.load_state_dict(best_model_param)
	single_test(opt, model, test_loader, loss_func, optimizer, scheduler)
