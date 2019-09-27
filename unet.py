# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from torch import autograd

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                    #返回1*1的池化结果
        self.fc = nn.Sequential( 
            nn.Linear(channel, channel // reduction, bias=False),  #W1=C/r*C （1,1,C/r)
            nn.ReLU(inplace=True),                                                     
            nn.Linear(channel // reduction, channel, bias=False),  #W2=C*C/r （1,1,C）
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #view 相同的数据 不一样的大小（size）
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)   #expend_as到和x一样的维度

class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.se1 = SELayer(64, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.se2 = SELayer(128, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.se3 = SELayer(256, 16)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.se4 = SELayer(512, 16)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)
	

    def forward(self,x):
        c1=self.conv1(x)
        #print(c1.size())
        c1=self.se1(c1)
        print(c1.size())
        p1=self.pool1(c1)

        c2=self.conv2(p1)
        c2=self.se2(c2)
        p2=self.pool2(c2)

        c3=self.conv3(p2)
        c3=self.se3(c3)
        p3=self.pool3(c3)

        c4=self.conv4(p3)
        c4=self.se4(c4)
        p4=self.pool4(c4)

        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out










