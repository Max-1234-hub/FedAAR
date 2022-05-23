# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:52:49 2021

@author: axmao2-c
"""

import torch
from torchvision.models import resnet18


x = torch.rand(1, 3, 224,224).cuda()

model = resnet18(pretrained=False).cuda()

for i in range(1000000000):
    for j in range(1000000000):
        for p in range(1000000000):
            for p in range(1000000000):
                for p in range(1000000000):
                    for p in range(1000000000):
                        for p in range(1000000000):
                            for p in range(1000000000):
                                for p in range(1000000000):
                                    for p in range(1000000000):
                                        for p in range(1000000000):
                                            for p in range(1000000000):
                                                for p in range(1000000000):
                                                    for p in range(1000000000):
                                                        for p in range(1000000000):
                                                            out = model(x)
                                                            loss = out.mean()
                                                            loss.backward()
           
