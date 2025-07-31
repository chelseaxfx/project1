# models/lora.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.r = r

    def forward(self, x):
        return F.linear(x, self.weight) + F.linear(x, self.B @ self.A)

class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=4, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lora_A = nn.Conv2d(in_channels, r, 1, bias=False)
        self.lora_B = nn.Conv2d(r, out_channels, 1, bias=False)
        self.alpha = 1.0  # 可调缩放系数

    def forward(self, x):
        return self.conv(x) + self.alpha * self.lora_B(self.lora_A(x))
