import torch
from torch import nn
from torch.nn import functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
    
        
        # (seq_length, d_model) pre initialized matrix
        self.pe = torch.zeros(seq_length, d_model)
        
        # create denominator constant of size (d_model / 2) for sine and cosine positions
        denominator = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        
        # define positions in order (seq_length, 1)
        positions = torch.arange(seq_length).unsqueeze(1)
        
        values = positions / denominator
        
        # select all even indices of pe for sine and all uneven for cosine
        self.pe[:, 0::2] = torch.sin(values)
        self.pe[:, 1::2] = torch.cos(values)
        
        # expand for batch dim (1, seq_length, d_model)
        self.pe = self.pe.unsqueeze(0).requires_grad_(False) # since the positional encodings are not trained
        
    def forward(self, embeddings: torch.Tensor):
        # input embeddings (batch_size, seq_length, d_model)
        seq_len = embeddings.shape[1]
        # Use only the needed portion of the positional encoding
        return self.dropout(embeddings + self.pe[:, :seq_len, :])
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        
        assert d_model % num_heads == 0, "Model dimensionality is not divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def attention(q, k, v, mask=None):
        d_k = q.shape[-1]
        scores = q @ k.transpose(-1, -2) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = F.softmax(scores, dim=-1) @ v
        return attention_scores
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.tensor, mask=None):
        # input dimensions: (batch_size, seq_length, model_dimension)
        shape = tuple(q.shape)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # split into different heads to perform multi head attention
        
        # (batch_size, seq_length, d_model) -> (batch_size, num_heads, seq_length, d_k)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
        
        attention_scores = self.attention(q, k, v, mask)
        
        # switch back (concat) to (batch_size, num_heads, seq_length, d_model) -> (batch_size, seq_length, d_model)
        attention_scores = attention_scores.transpose(-1, -2).contiguous().view(shape)
        
        
        return self.w_o(attention_scores)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048):
        super().__init__()
        
        self.input_to_hidden = nn.Linear(d_model, d_ff, bias=True)
        self.hidden_to_output = nn.Linear(d_ff, d_model, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.input_to_hidden(x)
        x = self.relu(x)
        x = self.hidden_to_output(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
    
    def forward(self, x, mask=None):
        residual = x
        x = self.attention(x, x, x, mask)
        x += residual
        # layer norm on all dimensions except batch
        x = F.layer_norm(x, x.shape[1:])
        
        residual = x
        x = self.feed_forward(x)
        
        x += residual
        # layer norm on all dimensions except batch
        x = F.layer_norm(x, x.shape[1:])
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int = 2048):
        super().__init__()
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
    def forward(self, x, mask=None):
        # iterate through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        return x
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        
        residual = x
        x = self.self_attention(x, x, x, mask=decoder_mask)
        x += residual
        x = F.layer_norm(x, x.shape[1:])
        
        residual = x
        x = self.cross_attention(x, encoder_output, encoder_output, mask=encoder_mask)
        x += residual
        x = F.layer_norm(x, x.shape[1:])
        
        residual = x
        x = self.feed_forward(x)
        x += residual
        x = F.layer_norm(x, x.shape[1:])
    
        return x
        

class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers, d_ff: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
    def forward(self, x, encoder_output, encoder_mask=None, decoder_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return x
    
        
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, seq_length: int, num_layers, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_length, dropout)
        
        self.encoder = Encoder(d_model, num_heads, num_layers, d_ff)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_ff)
        
        self.output = nn.Linear(d_model, tgt_vocab_size)

    
    def forward(self, src, tgt, encoder_mask=None, decoder_mask=None):

        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        encoder_output = self.encoder(src, mask=encoder_mask)
        
        
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        decoder_output = self.decoder(tgt, encoder_output, encoder_mask, decoder_mask)
        
        # final probabilities
        output = F.softmax(self.output(decoder_output), dim=-1)
        return output
        
        
        