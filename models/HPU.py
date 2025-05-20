import torch.nn as nn
import torch
class HPU(nn.Module):
    def __init__(self, input_channels=3, conv_kernels=[(5,1), (1,5)]):
        super().__init__()
        self.scale = input_channels**-0.5
        
        self.linear_o = nn.Linear(input_channels, input_channels)
        self.linear_p = nn.Linear(input_channels, input_channels)
        self.norm = nn.LayerNorm(input_channels)
        self.soft = nn.Softmax(-1)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.Q_proj = nn.Linear(input_channels, input_channels)
        self.K_proj = nn.Linear(input_channels, input_channels)
        self.sig = nn.Sigmoid()

        self.sc_ops = nn.ModuleList()
        for kernel in conv_kernels:
            padding = (kernel[0]//2, kernel[1]//2)
            self.sc_ops.append(nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=kernel, 
                         padding=padding, groups=input_channels),
                nn.GELU(),
                nn.Conv2d(input_channels, input_channels, 1),
                nn.BatchNorm2d(input_channels)
            ))
        
        self.conv_h = nn.Conv2d(input_channels*2, input_channels, 1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward_lpg(self, x):
        x_gap = self.gap(x) * x
        B, C, H, W = x_gap.shape
        x_gap = x_gap.view(B, C, -1).permute(0, 2, 1).contiguous()
        x_Q = self.Q_proj(x_gap)
        x_K = self.K_proj(x_gap)
        x_V = self.sig(x_Q) * x_K + x_gap
        return x_V.permute(0, 2, 1).contiguous().view(B, C, H, W)

    def forward_hpg(self, x):
        outputs = []
        for op in self.sc_ops:
            outputs.append(op(x))
        return self.conv_h(torch.cat(outputs, dim=1))

    def forward_pgc(self, prompt_l, prompt_h, x_ori):
        B, C, H, W = x_ori.shape
        x_V = self.linear_o(x_ori.view(B, C, -1).permute(0, 2, 1).contiguous())
        x_K = prompt_l.view(B, C, -1).permute(0, 2, 1).contiguous()
        x_Q = prompt_h.view(B, C, -1).permute(0, 2, 1).contiguous()
        
        x_attn = x_Q @ x_K.transpose(1, 2) * self.scale
        prompt = self.soft(x_attn) @ x_V
        p_norm = self.norm(self.linear_p(prompt) + x_V)
        return self.up(p_norm.permute(0, 2, 1).contiguous().view(B, C, H, W))

    def forward(self, x):
        x = self.pool(x)
        prompt_l = self.forward_lpg(x)
        prompt_h = self.forward_hpg(x)
        return self.forward_pgc(prompt_l, prompt_h, x)
