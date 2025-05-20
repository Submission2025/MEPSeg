import torch

import torch.nn as nn
from models.MFFE import MFFE
from models.EFR import EFR


class MSFA_Block(nn.Module):
    def __init__(self,in_channels,kernel,sample1=None,sample2=None):
        super().__init__()
        self.sample1=sample1
        self.sample2=sample2
        self.extract=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel,padding=kernel//2,groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self,x):
        if self.sample1!=None:
            x=self.sample1(x)
        x=self.extract(x)
        if self.sample2!=None:
            x=self.sample2(x)
        return x
    
    
class MSFA(nn.Module):
    def __init__(self, in_channels, kernel_list):
        super().__init__()
        
        block_configs = [
            (kernel_list[0], None, None),
            (kernel_list[1], None, None),
            (kernel_list[0], nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.MaxPool2d(2,2)),
            (kernel_list[1], nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.MaxPool2d(2,2)),
            (kernel_list[0], nn.MaxPool2d(2,2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            (kernel_list[1], nn.MaxPool2d(2,2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        ]
        
        self.blocks = nn.ModuleList([MSFA_Block(in_channels, k, s1, s2) for k, s1, s2 in block_configs])
        
        extract_layers = [
            nn.Conv2d(6*in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1)
        ]
        self.extract = nn.Sequential(*extract_layers)
    
    def forward(self, x):
        features = [block(x) for block in self.blocks]
        out = torch.cat(features, dim=1)
        return self.extract(out)



class EEU(nn.Module):
    def __init__(self,in_channels,kernel_list, wavelet, level, mode,dilation_values):
        super().__init__()
        self.msfa=MSFA(in_channels,kernel_list=kernel_list)
        self.mffe=MFFE(in_channels=in_channels, wavelet=wavelet, level=level, mode=mode)
        self.efr=EFR(in_channels=in_channels, 
                dilation_values=dilation_values)


    def forward(self,x):
        x=self.msfa(x)
        x=self.mffe(x)
        x=self.efr(x)
        return x


