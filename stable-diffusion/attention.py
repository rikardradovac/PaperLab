import torch
from torch import nn
import math
import torch.nn.functional as F



class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, d_emb: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.w_q = nn.Linear(d_emb, d_emb, bias=in_proj_bias)
        self.w_k = nn.Linear(d_emb, d_emb, bias=in_proj_bias)
        self.w_v = nn.Linear(d_emb, d_emb, bias=in_proj_bias)
        self.w_o = nn.Linear(d_emb, d_emb, bias=out_proj_bias)
        
        self.num_heads = num_heads
        self.d_head = d_emb // num_heads
        
    def forward(self, q, k, v, causal_mask=False):
        
        
        b, seq_length, d_emb = q.shape
        

        intermediate_shape = (b, seq_length, self.num_heads, self.d_head)
        
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        
        # (b, seq_len, num_heads, num_heads, d_head)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
        
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        