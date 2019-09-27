# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from torch import autograd
from torch.nn.modules.utils import _single, _pair, _triple

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class OctConv2d(nn.modules.conv._ConvNd):
    """Unofficial implementation of the Octave Convolution in the "Drop an Octave" paper.
    oct_type (str): The type of OctConv you'd like to use. ['first', 'A'] both stand for the the first Octave Convolution.
                    ['last', 'C'] both stand for th last Octave Convolution. And 'regular' stand for the regular ones.
    """
    
    def __init__(self,  in_channels, out_channels, kernel_size, stride=1, padding=1,  bias=True, alpha_in=0.5, alpha_out=0.5):
        
        if oct_type not in ('regular', 'first', 'last', 'A', 'C'):
            raise InvalidOctType("Invalid oct_type was chosen!")

        oct_type_dict = {'first': (0, alpha_out), 'A': (0, alpha_out), 'last': (alpha_in, 0), 'C': (alpha_in, 0), 
                         'regular': (alpha_in, alpha_out)}        

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        # TODO: Make it work with any padding
        padding = _pair(int((kernel_size[0] - 1) / 2))
        # padding = _pair(padding)
        #dilation = _pair(dilation)
        super(OctConv2d, self).__init__( in_channels, out_channels, kernel_size, stride, padding,  False, _pair(0), 1, bias)

        # Get alphas from the oct_type_dict
        self.oct_type = oct_type
        self.alpha_in, self.alpha_out = oct_type_dict[self.oct_type]
        
        self.num_high_in_channels = int((1 - self.alpha_in) * in_channels)
        self.num_low_in_channels = int(self.alpha_in * in_channels)
        self.num_high_out_channels = int((1 - self.alpha_out) * out_channels)
        self.num_low_out_channels = int(self.alpha_out * out_channels)

        self.high_hh_weight = self.weight[:self.num_high_out_channels, :self.num_high_in_channels, :, :].clone()
        self.high_hh_bias = self.bias[:self.num_high_out_channels].clone()

        self.high_hl_weight = self.weight[self.num_high_out_channels:, :self.num_high_in_channels, :, :].clone()
        self.high_hl_bias = self.bias[self.num_high_out_channels:].clone()

        self.low_lh_weight = self.weight[:self.num_high_out_channels, self.num_high_in_channels:, :, :].clone()
        self.low_lh_bias = self.bias[:self.num_high_out_channels].clone()

        self.low_ll_weight = self.weight[self.num_high_out_channels:, self.num_high_in_channels:, :, :].clone()
        self.low_ll_bias = self.bias[self.num_high_out_channels:].clone()

        self.high_hh_weight.data, self.high_hl_weight.data, self.low_lh_weight.data, self.low_ll_weight.data = \
        self._apply_noise(self.high_hh_weight.data), self._apply_noise(self.high_hl_weight.data), \
        self._apply_noise(self.low_lh_weight.data), self._apply_noise(self.low_ll_weight.data)

        self.high_hh_weight, self.high_hl_weight, self.low_lh_weight, self.low_ll_weight = \
        nn.Parameter(self.high_hh_weight), nn.Parameter(self.high_hl_weight), nn.Parameter(self.low_lh_weight), nn.Parameter(self.low_ll_weight)

        self.high_hh_bias, self.high_hl_bias, self.low_lh_bias, self.low_ll_bias = \
        nn.Parameter(self.high_hh_bias), nn.Parameter(self.high_hl_bias), nn.Parameter(self.low_lh_bias), nn.Parameter(self.low_ll_bias)
        

        self.avgpool = nn.AvgPool2d(2)
 
    def forward(self, x):
        if self.oct_type in ('first', 'A'):
            high_group, low_group = x[:, :self.num_high_in_channels, :, :], x[:, self.num_high_in_channels:, :, :]
        else:
            high_group, low_group = x

        high_group_hh = F.conv2d(high_group, self.high_hh_weight, self.high_hh_bias, self.stride,
                        self.padding,  self.groups)
        high_group_pooled = self.avgpool(high_group)

        if self.oct_type in ('first', 'A'):
            high_group_hl = F.conv2d(high_group_pooled, self.high_hl_weight, self.high_hl_bias, self.stride,
                        self.padding,  self.groups)
            high_group_out, low_group_out = high_group_hh, high_group_hl

            return high_group_out, low_group_out

        elif self.oct_type in ('last', 'C'):
            low_group_lh = F.conv2d(low_group, self.low_lh_weight, self.low_lh_bias, self.stride,
                            self.padding, self.groups)
            low_group_upsampled = F.interpolate(low_group_lh, scale_factor=2)
            high_group_out = high_group_hh + low_group_upsampled

            return high_group_out

        else:
            high_group_hl = F.conv2d(high_group_pooled, self.high_hl_weight, self.high_hl_bias, self.stride,
                        self.padding,  self.groups)
            low_group_lh = F.conv2d(low_group, self.low_lh_weight, self.low_lh_bias, self.stride,
                            self.padding, self.groups)
            low_group_upsampled = F.interpolate(low_group_lh, scale_factor=2)
            low_group_ll = F.conv2d(low_group, self.low_ll_weight, self.low_ll_bias, self.stride,
                            self.padding,  self.groups)
            
            high_group_out = high_group_hh + low_group_upsampled
            low_group_out = high_group_hl + low_group_ll

        return high_group_out, low_group_out

    @staticmethod
    def _apply_noise(tensor, mu=0, sigma=0.0001):
        noise = torch.normal(mean=torch.ones_like(tensor) * mu, std=torch.ones_like(tensor) * sigma)

        return tensor + noise


class OctReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu_h, self.relu_l = nn.ReLU(inplace), nn.ReLU(inplace)

    def forward(self, x):
        h, l = x

        return self.relu_h(h), self.relu_l(l)


class OctMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,  return_indices=False, ceil_mode=False):
        super().__init__()
        self.maxpool_h = nn.MaxPool2d(kernel_size, stride=None, padding=0,  return_indices=False, ceil_mode=False)
        self.maxpool_l = nn.MaxPool2d(kernel_size, stride=None, padding=0,  return_indices=False, ceil_mode=False)

    def forward(self, x):
        h, l = x

        return self.maxpool_h(h), self.maxpool_l(l)


class Error(Exception):
    """Base-class for all exceptions rased by this module."""


class InvalidOctType(Error):
    """There was a problem in the OctConv type."""

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)                    #返回1*1的池化结果
#         self.fc = nn.Sequential( 
#             nn.Linear(channel, channel // reduction, bias=False),  #W1=C/r*C （1,1,C/r)
#             nn.ReLU(inplace=True),                                                     
#             nn.Linear(channel // reduction, channel, bias=False),  #W2=C*C/r （1,1,C）
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c) #view 相同的数据 不一样的大小（size）
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)   #expend_as到和x一样的维度

class Unet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unet, self).__init__()

        #self.conv1 = DoubleConv(in_channels, 64)

	self.conv_1 = OctConv2d('first', in_channels=3, out_channels=64, kernel_size=3)
        self.r1 = OctReLU()
        self.conv_2 = OctConv2d('last', in_channels=64, out_channels=64, kernel_size=3)
        self.r2 = OctReLU()

        #self.se1 = SELayer(64, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        #self.se2 = SELayer(128, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        #self.se3 = SELayer(256, 16)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        #self.se4 = SELayer(512, 16)
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
        self.conv10 = nn.Conv2d(64,out_channels, 1)
	

    def forward(self,x):
        #c1=self.conv1(x)
        #c1=self.se1(c1)
	c_1 = self.con_1(x)
	c1 = self.con_2(C_1)


        p1=self.pool1(c1)

        c2=self.conv2(p1)
        #c2=self.se2(c2)
        p2=self.pool2(c2)

        c3=self.conv3(p2)
        #c3=self.se3(c3)
        p3=self.pool3(c3)

        c4=self.conv4(p3)
        #c4=self.se4(c4)
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










