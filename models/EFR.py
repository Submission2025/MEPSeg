import torch

import torch.nn as nn


import torchvision

import torch.nn as nn

class DirectionOffsets(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.offsets = nn.ModuleList()
        
        offset_confs = [
            {'kernel_size': (1, 15), 'padding': (0, 7)},
            {'kernel_size': (15, 1), 'padding': (7, 0)},
            {'kernel_size': 3, 'padding': 1}
        ]
        
        for conf in offset_confs:
            self.offsets.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=conf['kernel_size'],
                        padding=conf['padding'],
                        groups=in_channels
                    ),
                    nn.BatchNorm2d(num_features=in_channels)
                )
            )
            
        self.balance = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * kernel_size * kernel_size,
                kernel_size=1
            ),
            nn.BatchNorm2d(num_features=2 * kernel_size * kernel_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_sum = sum(offset_module(x) for offset_module in self.offsets)
        
        return self.balance(offset_sum)

    
class EFR(nn.Module):
    def __init__(self, 
                in_channels: int, 
                dilation_values: list,
                padding_values: list = None) -> None:
        super().__init__()
        
        if padding_values is None:
            padding_values = dilation_values.copy()
        
        dilation_confs = [
            {'padding': p, 'dilation': d} 
            for p, d in zip(padding_values, dilation_values)
        ]
        
        self.offset = DirectionOffsets(in_channels)
        self.deforms = nn.ModuleList()
        
        for conf in dilation_confs:
            self.deforms.append(
                torchvision.ops.DeformConv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=conf['padding'],
                    groups=in_channels,
                    dilation=conf['dilation'],
                    bias=False
                )
            )
            
        self.balance = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(num_features=in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = self.offset(x)
        deform_sum = sum(deform(x, offsets) for deform in self.deforms)
        return self.balance(deform_sum) * x
