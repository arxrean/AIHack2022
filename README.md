```shell
CUDA_VISIBLE_DEVICES=1 python main_3d_unet.py --num_workers 0 --gpu --epoches 40 --gpu --pad_size 64 --plot --batch_size 32 --loss05 1 --loss_peak 0.02 --unet_atten --activation sigmoid --log
```

