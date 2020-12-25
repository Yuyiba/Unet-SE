# -*- coding:utf-8 -*-
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from tensorboardX import SummaryWriter
from PIL import Image
import os.path
import glob
from pylab import *
from tqdm import tqdm

if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
if not os.path.exists("result"):
        os.makedirs("result")
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

#参数解析
parse=argparse.ArgumentParser()

def train_model(model, criterion, optimizer, dataload, num_epochs=100):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 40)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y,x_path in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))

            writer.add_scalar('loss',loss.item(),step+epoch*111)
            writer.add_scalar('train_loss',epoch_loss / 111,epoch)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
#save epoch
        torch.save(model.state_dict(), './checkpoints/weights_%d.pth' % epoch)

    
    #writer.add_scalar('acc',accuracy,epoch)

    return model

#losses_his=[]

writer = SummaryWriter('/home/yus/Documents/Unet-SE-master/log/')


#训练模型
def train():
    model = Unet(3,1).to(device)
    #model.load_state_dict(torch.load('./checkpoints/weights_39.pth'))

    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)   #4
    train_model(model, criterion, optimizer, dataloaders)


#显示模型的输出结果
def test():
    model = Unet(3, 1).to(device)
    model.load_state_dict(torch.load(args.ckp))
    liver_dataset = LiverDataset("data/test", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    with torch.no_grad():
        for x, _ ,x_path in tqdm(dataloaders):
            x_path=str(x_path).split("/")
            x = x.to(device)
            y=model(x)
            img_numpy = y[0].cpu().float().numpy()
            img_numpy = (np.transpose(img_numpy, (1, 2, 0)))
            img_numpy = (img_numpy >= 0.5) * 255
            img_out=img_numpy.astype(np.uint8)
            imgs = transforms.ToPILImage()(img_out)
            imgs.save('result/'+x_path[2][:-3])




if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=4) #6
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train()
    elif args.action=="test":
        test()
