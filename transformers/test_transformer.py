import pytest
import torch
import torch.nn as nn
import math
from transformers.transformer import (
    Transformer,
    MultiHeadAttention,
    FeedForward,
    PositionalEncoding
)

@pytest.fixture
def transformer_config():
    return {
        'src_vocab_size': 1000,
        'tgt_vocab_size': 1200,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'seq_length': 100
    }

def test_feed_forward():
    ff = FeedForward(d_model=512, d_ff=2048)
    x = torch.randn(2, 10, 512)  # (batch_size, seq_len, d_model)
    
    output = ff(x)
    
    assert output.shape == x.shape
    assert not torch.allclose(output, x)  # Output should be transformed

def test_multi_head_attention():
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)  # (batch_size, seq_len, d_model)
    
    # Test self-attention
    output = mha(x, x, x)
    assert output.shape == x.shape
    
    # Test with mask - shape should be (batch_size, num_heads, seq_len, seq_len)
    mask = torch.ones(2, 8, 10, 10)  # Changed from (2, 1, 10, 10)
    mask[:, :, :, 5:] = 0  # Mask out positions after index 5
    output_masked = mha(x, x, x, mask=mask)
    assert output_masked.shape == x.shape
    assert not torch.allclose(output, output_masked)

def test_positional_encoding():
    pe = PositionalEncoding(d_model=512, seq_length=100)
    x = torch.randn(2, 10, 512)
    
    output = pe(x)
    assert output.shape == x.shape
    assert not torch.allclose(output, x)

def test_transformer_shape(transformer_config):
    transformer = Transformer(**transformer_config)
    
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = torch.randint(0, transformer_config['src_vocab_size'], (batch_size, src_seq_len))
    tgt = torch.randint(0, transformer_config['tgt_vocab_size'], (batch_size, tgt_seq_len))
    
    output = transformer(src, tgt)
    
    expected_shape = (batch_size, tgt_seq_len, transformer_config['tgt_vocab_size'])
    assert output.shape == expected_shape

def test_transformer_inference(transformer_config):
    transformer = Transformer(**transformer_config)
    transformer.eval()  # Set to evaluation mode
    
    # Test single sequence
    src = torch.randint(0, transformer_config['src_vocab_size'], (1, 5))
    tgt = torch.randint(0, transformer_config['tgt_vocab_size'], (1, 1))  # Start token
    
    with torch.no_grad():
        output = transformer(src, tgt)
    
    assert output.shape == (1, 1, transformer_config['tgt_vocab_size'])
    assert torch.isfinite(output).all()  # Check for NaN/inf values

@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("seq_length", [10, 50, 100])
def test_transformer_different_sizes(transformer_config, batch_size, seq_length):
    transformer = Transformer(**transformer_config)
    
    src = torch.randint(0, transformer_config['src_vocab_size'], (batch_size, seq_length))
    tgt = torch.randint(0, transformer_config['tgt_vocab_size'], (batch_size, seq_length))
    
    output = transformer(src, tgt)
    
    expected_shape = (batch_size, seq_length, transformer_config['tgt_vocab_size'])
    assert output.shape == expected_shape