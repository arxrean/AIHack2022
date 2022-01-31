# Train

Use command below to run our final model.

```shell
python main_3d_mt.py --num_workers 0 --gpu --epoches 50 --pad_size 64 --batch_size 32 --cin 2 --h5_times 39 --aug --unet_atten --activation sigmoid --lr 1e-3 --loss05 0.5 --loss_peak 0.01 --log
```

Use command below to test the trained model. You need to first download the trained model from your own wandb website. Change the last three parameters to adjust tile size (1 for no tile), test samples and test time steps, respectively. 

```shell
python main_3d_mt_test.py --num_workers 0 --gpu --epoches 50 --pad_size 64 --batch_size 32 --cin 2 --h5_times 39 --aug --unet_atten --activation sigmoid --lr 1e-3 --loss05 0.5 --loss_peak 0.01 --tile 1 --test_samples 5 --test_steps 100
```

To generate some of the physical correctness figures, run utils.py.
In order to do so, change roots in <if __name__=="__main__"> to locate generated data produced from main_3d_mt_test.py (above code) and then run the following.
  
```shell
python utils.py  
```  
