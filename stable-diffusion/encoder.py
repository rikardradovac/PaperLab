import torch
from torch import nn
import torch.nn.functional as F
from decoder import AttentionBlock, ResidualBlock


class VAE(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            ResidualBlock(128, 128),
            
            ResidualBlock(128, 128),
            
            nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0),
            
            
            ResidualBlock(128, 256),
            
            ResidualBlock(256, 256),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            ResidualBlock(256, 512),
            
            ResidualBlock(512, 512),
            
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            ResidualBlock(512, 512),
            
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            AttentionBlock(512),
            
            nn.Silu(),
            
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
            
             
    
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        #x: (b, channel, height, width)
        
        for module in self:
            if getattr(module, "padding", None) == (2, 2):
                x = F.pad(x, (1, 0, 1, 0))
                
            x = module(x)
            
        # two tensors of size (b, 4, height/8, width/8 )
        mean, log_var = torch.chunk(x, 2, dim=1)
        
        log_var = torch.clamp(log_var, -30, 20)
        variance = log_var.exp()
        std = variance.sqrt()
        
        
        x = mean + std * noise
        
        x *= 0.18215
        