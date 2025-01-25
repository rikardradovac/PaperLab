import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention



class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x):
        
        residual = x
        
        m, c, h, w = x.shape
        
        x = x.view(m, c, h * w)
        
        x = x.transpose(-1, -2)
        
        x = self.attention(x)
        
        x = x.transpose(-1, -2)
        
        x = x.view((m, c, h, w))
        
        return x + residual

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.group_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.group_norm_1 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x: torch.Tensor):
        
        residual = x
        
        x = self.group_norm_1(x)
        
        x = F.silu()
        
        x = self.conv_1(x)
        
        x = self.group_norm_2(x)
        
        x = F.silu()
        
        x = self.group_norm_2(x)
        
        x = self.conv_2(x)
        
        
        return x + self.residual_layer(residual)