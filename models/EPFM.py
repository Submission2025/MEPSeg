import torch
from torch import nn
from models.EEU import EEU
from models.HPU import HPU
import torch
import torch.nn as nn

class EPFM(nn.Module):
    def __init__(self, in_channels, out_channels, sample, kernel_list, wavelet, level, mode,dilation_values, conv_kernels, up=True):
        super().__init__()
        
        self.eeu = EEU(in_channels, kernel_list=kernel_list, wavelet=wavelet, level=level, mode=mode,dilation_values=dilation_values)
        self.hpu = HPU(in_channels,conv_kernels=conv_kernels)
        
        mlp_layers = [nn.BatchNorm2d(in_channels*2)]
        conv_channels = [(in_channels*2, out_channels), (out_channels, out_channels)]
        for i, (cin, cout) in enumerate(conv_channels):
            mlp_layers.append(nn.Conv2d(cin, cout, 1))
            if i < len(conv_channels)-1:
                mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.BatchNorm2d(out_channels))
        self.mlp = nn.Sequential(*mlp_layers)
        
        self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if (sample and up) else \
                      nn.MaxPool2d(2, stride=2) if (sample and not up) else None

    def forward(self, x):
        x_eeu = self.eeu(x)
        x_hpu = self.hpu(x)
        x_cat = torch.cat([x_eeu, x_hpu], dim=1)
        x = self.mlp(x_cat)
        return self.sample(x) if self.sample else x

