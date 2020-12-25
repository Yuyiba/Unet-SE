# Unet-SE backbone 

## 数据准备
项目文件分布如下
```
  --project
  	main.py
  	data
       --train
       --val
```


## Train & Test
```
#### train
python main.py train
#### test
python main.py test --ckp=weights_11.pth 

#### tensorboard
tensorboard --logdir=/home/yus/Documents/u_net_liver-master/tensorboard/

```


