# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:52:49 2021

@author: axmao2-c
"""

import torch.nn as nn
import torch

class CaNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        self.conv1 = BasicBlock_b(in_channels=1, out_channels=8)
        self.conv2 = BasicBlock_b(in_channels=8, out_channels=16)
        self.conv3 = BasicBlock_b(in_channels=16, out_channels=32)
        self.conv4 = BasicBlock_b(in_channels=32, out_channels=64)
        
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))
        self.CA_C_s = BasicBlock()
        
        self.conv5_acc = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv5_gyr = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        xa = x[:,:,:,0:3]
        xg = x[:,:,:,3:6]
        
        xa = xa.permute(0,1,3,2)
        xg = xg.permute(0,1,3,2)
        output_x, output_y = self.conv1(xa, xg)
        output_x = self.maxpool(output_x)
        output_y = self.maxpool(output_y)
        #output_x, output_y, att_map_acc, att_map_gyr = self.CA_C_s(output_x, output_y)
        output_x, output_y = self.conv2(output_x, output_y)
        output_x = self.maxpool(output_x)
        output_y = self.maxpool(output_y)
        #output_x, output_y, att_map_acc, att_map_gyr = self.CA_C_s(output_x, output_y)
        output_x, output_y = self.conv3(output_x, output_y)
        output_x = self.maxpool(output_x)
        output_y = self.maxpool(output_y)
        #output_x, output_y, att_map_acc, att_map_gyr = self.CA_C_s(output_x, output_y)
        output_x, output_y = self.conv4(output_x, output_y)
        output_x, output_y, att_map_acc, att_map_gyr = self.CA_C_s(output_x, output_y)
        # print(output_x.size())
    
        output_x = self.conv5_acc(output_x)
        output_y = self.conv5_gyr(output_y)
        output_cat = torch.cat((output_x,output_y), 1)
        
        output_cat = self.avg_pool(output_cat) #[batch_size, num_filters, 1, 1]
        output_cat = output_cat.view(output_cat.size(0), -1) #[batch_size, num_filters]
        output = self.fc(output_cat)
        # print(output.size())
    
        return output, output_cat


class BasicBlock_b(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #residual function
        self.residual_function_acc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_function_gyr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut_acc = nn.Sequential()
        self.shortcut_gyr = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if in_channels != out_channels:
            self.shortcut_acc = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,1)),
                nn.BatchNorm2d(out_channels)
            )
            self.shortcut_gyr = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x_acc, x_gyr):
        out_acc = self.residual_function_acc(x_acc)
        out_gyr = self.residual_function_gyr(x_gyr)

        acc_output = nn.ReLU(inplace=True)(out_acc + self.shortcut_acc(x_acc))
        gyr_output = nn.ReLU(inplace=True)(out_gyr + self.shortcut_gyr(x_gyr))
        
        return acc_output, gyr_output



class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.FM_A = Feature_map_att(1,16)

    def forward(self, x_acc, x_gyr):
        out_acc, out_gyr, att_map_acc, att_map_gyr = self.FM_A(x_acc, x_gyr)
        acc_output = out_acc + x_acc
        gyr_output = out_gyr + x_gyr
        
        return acc_output, gyr_output, att_map_acc, att_map_gyr


#cross attention by channel-wise
class Feature_map_att(nn.Module):
    def __init__(self, input_channel=1, middle_channel=16): #dim_acc, dim_gyr= 64
        super().__init__()
        
        self.conv_combination1 = nn.Sequential(
            nn.Conv2d(2*input_channel, middle_channel, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True))
        self.conv_acc = nn.Sequential(
            nn.Conv2d(middle_channel, input_channel, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(input_channel),
            nn.Sigmoid())
        self.conv_gyr = nn.Sequential(
            nn.Conv2d(middle_channel, input_channel, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(input_channel),
            nn.Sigmoid())
        
    def forward(self, f_acc, f_gyr):
        b, c, _, w = f_acc.size()
        squeeze_array = []
        for tensor in [f_acc, f_gyr]:
            tview = torch.mean(tensor, dim=1, keepdim=True, out=None)
            squeeze_array.append(tview)
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.conv_combination1(squeeze)
        # excitation = self.conv_combination2(excitation)
        acc_out = self.conv_acc(excitation)
        gyr_out = self.conv_gyr(excitation)
      
        return f_acc * acc_out.expand_as(f_acc), f_gyr * gyr_out.expand_as(f_gyr), acc_out, gyr_out

net = CaNet(6)
x = torch.rand(4,1,200,6)
output, output_cat = net(x)

# for key in net.state_dict().keys():
#     print(key,net.state_dict()[key].shape)
    # print(key)
# layer = []
# for i in list(net.state_dict().keys()):
#     idx = i.index('.')
#     layer.append(i[:idx])
# print(layer)
# labels = sorted(set(layer),key=layer.index)
# print(labels)

# num_l = []
# for l in labels:
#     length_l = 0
#     print(l)
#     for key in net.state_dict().keys():
#         if (l in key) and ('num_batches_tracked' not in key):
#             dims = net.state_dict()[key].shape
#             length_l += dims.numel()
#     num_l.append(length_l)
# print(num_l)

# print(list(net.state_dict().keys())[30][0:idx])