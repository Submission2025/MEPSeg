import torch.nn as nn



from models.EPFM import EPFM
class MEPSeg(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_list, wavelet, level, mode,dilation_values, conv_kernels):
        super().__init__()
        
        self.en_layers = nn.ModuleList([
            EPFM(out_channels[i], out_channels[i+1], sample=True, up=False, kernel_list=kernel_list, wavelet=wavelet, level=level, mode=mode, dilation_values=dilation_values, conv_kernels=conv_kernels)
            for i in range(len(out_channels)-1)
        ])
        
        self.de_layers = nn.ModuleList([
            EPFM(out_channels[i+1], out_channels[i], sample=True, up=True, kernel_list=kernel_list, wavelet=wavelet, level=level, mode=mode, dilation_values=dilation_values, conv_kernels=conv_kernels)
            for i in reversed(range(len(out_channels)-1))
        ])
        
        self.patch_conv = nn.Sequential(
            nn.Conv2d(input_channels, out_channels[0], 3, padding=1),
            nn.BatchNorm2d(out_channels[0])
        )
        
        self.ph = PH(out_channels)
    
    def forward(self, x):
        x = self.patch_conv(x)
        
        enc_outputs = []
        for en_layer in self.en_layers:
            x = en_layer(x)
            enc_outputs.append(x)
        
        dec_outputs = []
        x_dec = enc_outputs[-1] if enc_outputs else x
        for i, de_layer in enumerate(self.de_layers):
            x_dec = de_layer(x_dec)
            dec_outputs.append(x_dec)
            if i < len(self.de_layers)-1:
                x_dec += enc_outputs[-i-2]
        
        return self.ph(dec_outputs)



class PH_Block(nn.Module):
    def __init__(self, in_channels, scale_factor=1):
        super().__init__()
        if scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample = None
        self.pro = nn.Conv2d(in_channels, 1, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.pro(x)
        x = self.sig(x)
        return x

class PH(nn.Module):
    def __init__(self, in_channels=[12,24,36,48], scale_factor=[1,2,4,8]):
        super().__init__()
        self.ph_blocks = nn.ModuleList(
            [PH_Block(inc, sf) for inc, sf in zip(in_channels, scale_factor)]
        )
        
        for idx, block in enumerate(self.ph_blocks, 1):
            setattr(self, f'ph{idx}', block)
            
    def forward(self, x):
        outputs = []
        for i in range(len(self.ph_blocks)):
            input_tensor = x[-(i+1)]  
            outputs.append(self.ph_blocks[i](input_tensor))
        
        return outputs
