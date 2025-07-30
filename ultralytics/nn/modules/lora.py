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
