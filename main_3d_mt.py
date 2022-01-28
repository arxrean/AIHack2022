import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fastai.vision.all import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from torch.optim import Adam, AdamW

import wandb
from main_3d import (Conv3dCustom, LeftModel, data_split, get_loss_func,
                     get_parser, unpad_input)
from unet import UNet


# h5 later
class RegDataset(torch.utils.data.Dataset):
    def __init__(self, opt, data, mode="train"):
        self.opt = opt
        self.mode = mode
        self.data = data

        self.h5s = [h5py.File(opt.h5_path, "r")["time{}".format(i)] for i in range(10)]

        if opt.data_norm:
            self.mean, self.std = (
                self.data.reshape(-1).mean(),
                self.data.reshape(-1).std(),
            )

        if mode == "train" and self.opt.train_frac < 1.0:
            self.data = self.data[
                np.random.choice(
                    len(self.data), int(len(self.data) * self.opt.train_frac)
                )
            ]

        if self.opt.log:
            wandb.config.update({self.mode: len(self.data)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time, time_idx = divmod(self.data[idx], self.opt.h5_size)
        prev = self.h5s[time + int(self.opt.h5_times[0]) - 2][time_idx]
        curr = self.h5s[time + int(self.opt.h5_times[0]) - 1][time_idx]
        label = self.h5s[time + int(self.opt.h5_times[0])][time_idx]
        curr = np.stack((prev, curr))

        if self.opt.data_norm:
            curr = (curr - self.mean) / self.std
            label = (label - self.mean) / self.std

        img_size, label_size = curr.shape[-1], label.shape[-1]
        assert img_size == label_size
        if self.opt.aug and self.mode == "train":
            dims = sorted(np.random.choice(3, np.random.randint(0, 4), replace=False))
            _dims = [x + 1 for x in dims]
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
        outs, labels = unpad_input(opt, model(images), labels, crop_sizes, "train")
        # get_structure_factor(labels.cpu().numpy()[0, 0])
        loss_mse, loss05, loss_peak = loss_func(opt, outs, labels)
        if opt.log:
            wandb.log({"train_loss_mse": loss_mse.item()})
        loss = loss_mse
        if opt.loss05 != 0:
            loss += loss05
        if opt.loss_peak != 0:
            loss += loss_peak
        if opt.time_weight:
            loss *= 1 + times.float().mean() / 10

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if opt.log:
            wandb.log({"train_loss": loss.item()})
            if opt.loss05:
                wandb.log({"train_loss05": loss05.item()})
            if opt.loss_peak:
                wandb.log({"train_loss_peak": loss_peak.item()})

    return model


def single_val(opt, model, loader, loss_func, optimizer, scheduler):
    model = model.eval()
    _loss = 0
    with torch.no_grad():
        for step, pack in enumerate(loader):
            images, labels, crop_sizes, times = pack
            if opt.gpu:
                images, labels = images.float().cuda(), labels.float().cuda()
            outs, labels = unpad_input(opt, model(images), labels, crop_sizes, "val")
            loss_mse, loss05, loss_peak = loss_func(opt, outs, labels)
            if opt.log:
                wandb.log({"train_loss_mse": loss_mse.item()})
            loss = loss_mse
            if opt.loss05 != 0:
                loss += loss05
            if opt.loss_peak != 0:
                loss += loss_peak
            if opt.time_weight:
                loss *= 1 + times.float().mean() / 10

            _loss += loss.item()
            if opt.log:
                wandb.log({"val_loss": loss.item()})
                if opt.loss05:
                    wandb.log({"val_loss05": loss05.item()})
                if opt.loss_peak:
                    wandb.log({"val_loss_peak": loss_peak.item()})

    return _loss / len(loader)


def single_test(opt, model, loader, loss_func, optimizer, scheduler):
    model = model.eval()
    inputs, preds, gts = [], [], []
    with torch.no_grad():
        for step, pack in enumerate(loader):
            images, labels, crop_sizes, times = pack
            if opt.gpu:
                images, labels = images.float().cuda(), labels.float().cuda()
            outs, labels = unpad_input(opt, model(images), labels, crop_sizes, "test")
            inputs.append(images.cpu().numpy())
            preds.append(outs.cpu().numpy())
            gts.append(labels.cpu().numpy())
    if opt.data_norm:
        inputs = (np.concatenate(inputs, 0) * test_set.std) + test_set.mean
        preds = (np.concatenate(preds, 0) * test_set.std) + test_set.mean
        gts = (np.concatenate(gts, 0) * test_set.std) + test_set.mean
    else:
        inputs = np.concatenate(inputs, 0)
        preds = np.concatenate(preds, 0)
        gts = np.concatenate(gts, 0)

    mse = mean_squared_error(gts.reshape(-1), preds.reshape(-1))
    rmse = mean_squared_error(gts.reshape(-1), preds.reshape(-1), squared=False)
    mae = mean_absolute_error(gts.reshape(-1), preds.reshape(-1))
    r2 = r2_score(gts.reshape(-1), preds.reshape(-1))

    print("MSE: {} RMSE: {} MAE: {} R2: {}".format(mse, rmse, mae, r2))

    if opt.log:
        wandb.config.update({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    if opt.log and opt.plot:
        inputs = inputs[:, 0]
        preds = preds[:, 0]
        gts = gts[:, 0]
        plots = np.random.choice(range(len(inputs)), opt.plot_num)
        # plt = plot_pairs(opt, inputs[plots], preds[plots], gts[plots])



class LitModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(opt, attention=opt.unet_atten, in_channels=opt.cin)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):

        images, labels, crop_sizes, times = batch
        # if opt.gpu:
        #     images, labels = images.float().cuda(), labels.float().cuda()
        outs, labels = unpad_input(opt, self.model(images), labels, crop_sizes, "train")
        loss = nn.MSELoss()(outs, labels)
        
        # # get_structure_factor(labels.cpu().numpy()[0, 0])
        # loss_mse, loss05, loss_peak = get_loss_func(opt, outs, labels)
        
        # # Log MSE before loss gets modified.
        # if opt.log:
        #     wandb.log({"train_loss_mse": loss_mse.item()})

        # loss = loss_mse
        # if opt.loss05 != 0:
        #     loss += loss05
        # if opt.loss_peak != 0:
        #     loss += loss_peak
        # if opt.time_weight:
        #     loss *= 1 + times.float().mean() / 10

        return loss

    def validation_step(self, batch, batch_idx):
        
        images, labels, crop_sizes, times = batch

        outs, labels = unpad_input(opt, model(images), labels, crop_sizes, "val")
        loss = nn.MSELoss()(outs, labels)
        
        # loss_mse, loss05, loss_peak = get_loss_func(opt, outs, labels)
        # # Log MSE before loss gets modified.
        # if opt.log:
        #     wandb.log({"train_loss_mse": loss_mse.item()})
        
        # loss = loss_mse
        # if opt.loss05 != 0:
        #     loss += loss05
        # if opt.loss_peak != 0:
        #     loss += loss_peak
        # if opt.time_weight:
        #     loss *= 1 + times.float().mean() / 10

        return loss



    


if __name__ == "__main__":
    opt = get_parser()
    if opt.log:
        wandb.init(project="AIHack", name=opt.name)
        wandb.config.update(opt)

    data = np.arange(opt.h5_size * (int(opt.h5_times[1]) - int(opt.h5_times[0]) + 1))

    train, val, test = data_split(opt, data)

    train_set, val_set, test_set = (
        RegDataset(opt, train, "train"),
        RegDataset(opt, val, "val"),
        RegDataset(opt, test, "test"),
    )
    if opt.data_norm:
        val_set.mean, val_set.std = train_set.mean, train_set.std
        test_set.mean, test_set.std = train_set.mean, train_set.std
    train_loader, val_loader, test_loader = (
        DataLoader(
            train_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            drop_last=True,
        ),
        DataLoader(
            val_set,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            drop_last=True,
        ),
        DataLoader(
            test_set,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
        ),
    )

    # model = UNet(opt, attention=opt.unet_atten, in_channels=opt.cin)

    model = LitModel()
    trainer = Trainer(gpus=1, benchmark=True)
    trainer.fit(model, train_loader, val_loader)

    
    # if opt.gpu:
    #     model = torch.nn.DataParallel(model)

    # optimizer = AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # total_training_steps = len(train_loader) * opt.epoches
    # warmup_steps = total_training_steps // opt.warm_up_split
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_training_steps,
    # )

    # best_model_param, best_loss = None, 1e5
    # for epoch in range(opt.epoches):
    #     model = single_train(
            # opt, model, train_loader, get_loss_func, optimizer, scheduler
    #     )
    #     val_loss = single_val(
    #         opt, model, val_loader, get_loss_func, optimizer, scheduler
    #     )
    #     if val_loss < best_loss:
    #         best_model_param = model.state_dict()
    #         best_loss = val_loss
    #     if opt.log:
    #         wandb.log({"epoch_loss": val_loss})

    # model.load_state_dict(best_model_param)
    # single_test(opt, model, test_loader, get_loss_func, optimizer, scheduler)
    # torch.save(
    #     model.cpu().state_dict(), os.path.join(wandb.run.dir, "{}.h5".format(opt.name))
    # )
