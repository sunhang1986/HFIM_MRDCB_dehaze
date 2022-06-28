import torch.nn as nn
import torch

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return y

class Meu_module(nn.Module):
    def __init__(self,low_c,high_c):
        super(Meu_module, self).__init__()
        modules_ca = []
        modules_pa = []
        self.low_conv = nn.Sequential(nn.Conv2d(low_c,low_c,kernel_size=3, stride=1, padding=1))
        self.high_conv = nn.Sequential(nn.ConvTranspose2d(high_c,low_c, kernel_size=3,stride=2,padding=1,output_padding=1))
        modules_ca.append(CALayer(channel=low_c))
        modules_pa.append(PALayer(channel=low_c))
        self.ca = nn.Sequential(*modules_ca)
        self.pa = nn.Sequential(*modules_pa)

    def forward(self,low_level,high_level):
        low = self.low_conv(low_level)
        high = self.high_conv(high_level)
        low_pa = self.pa(low)
        high_ca = self.ca(high)
        a = low * high_ca
        b = high * low_pa
        c = a+b
        return c

