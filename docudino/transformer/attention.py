import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, head_size: int):
        """
        Creates a new `nn.Module` that implements scaled dot-product attention
        
        Args:
            d_model (int): The dimensionality of the model
            head_size (int): the size of each head
        """
        
        super().__init__()
        
        # store parameters
        self.head_size = head_size
        
        # create projections
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
    
    def forward(self, x: torch.Tensor):
        # collect query, key, and value
        Q: torch.Tensor = self.query(x)
        K: torch.Tensor = self.key(x)
        V: torch.Tensor = self.value(x)
        
        # calculate attention score
        score = Q @ K.transpose(-2, -1)
        
        # scaling
        attn = score / (self.head_size ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        
        # return the weighted sum of values
        return attn @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        Creates a new `nn.Module` that implements multi-head attention by running multiple
        instances of `AttentionHead` in sequence.
        
        Args:
            d_model (int): The dimensionality of the model
            n_heads (int): The number of heads to create
        """
        
        super().__init__()
        
        # store parameters
        assert d_model % n_heads == 0, "Model dimensionality must be divisible by the number of heads"
        self.head_size = d_model // n_heads
        
        # concatenation projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # create list of heads
        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])
    
    def forward(self, x: torch.Tensor):
        # combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        
        # learnable projection
        out = self.W_o(out)
        
        return out