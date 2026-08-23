import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, qkv_bias: bool = True):
        """
        Creates a new `nn.Module` that implements multi-head attention through a unified qkv projection
        
        Args:
            d_model (int): The dimensionality of the model
            n_heads (int): The number of heads to create
            qkv_bias (bool): If true, the `qkv` projection will have a bias term
        """
        
        super().__init__()
        
        assert d_model % n_heads == 0, "Model dimensionality must be divisible by the number of heads"
        
        # store parameters
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        
        # create projections
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        
        # split qkv projection
        qkv: torch.Tensor = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # calculate attention
        attn: torch.Tensor = (Q @ K.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = (attn @ V)
        
        attn = attn.transpose(1, 2).reshape(B, N, C)
        
        # learnable projection
        attn = self.W_o(attn)
        
        return attn