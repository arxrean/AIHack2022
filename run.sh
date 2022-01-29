#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main_3d_mt.py --train_frac 0.1 --batch_size 28 --pad_size 64 --gpu --aug --unet_atten --epoches 20 --warm_up_split 5 --lr 1e-3 --weight_decay 0 --loss05 0.5 --loss_peak 0.0
CUDA_VISIBLE_DEVICES=1 python main_3d_mt.py --train_frac 0.1 --batch_size 28 --pad_size 64 --gpu --aug --unet_atten --epoches 20 --warm_up_split 5 --lr 1e-3 --weight_decay 5e-5 --loss05 1 --loss_peak 0.04
