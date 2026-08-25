import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .positional_encoding import LearnedPositionalEncoding
from .heads import DINOHead

class PatchEmbeddings(nn.Module):
    """
    A Neural Network module responsible for creating linear embeddings of patches in an input image.
    """
    
    def __init__(self, d_model: int, patch_size: int, n_channels: int):
        """
        Creates a new `nn.Module` that embeds patches from an image.
        
        Args:
            d_model (int): The dimensionality of the model
            patch_size (int): The size of the patches (square)
            n_channels (int): The number of channels passed into the transformer
        """
        super().__init__()
        
        # store parameters
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        # create the patch splitter
        self.linear_proj = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
    
    def forward(self, x: torch.Tensor):
        """
        Runs a forward pass of the patch embedder module. A few important values:
          - B: Batch Size
          - C: Color Channels
          - H: Image Height
          - W: Image Width
          - P_col: Patch columns
          - P_row: Patch rows
        """
        
        # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = self.linear_proj(x)
        
        # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.flatten(2)
        
        # (B, d_model, P) -> (B, P, d_model)
        x = x.transpose(1, 2)
        
        return x

class TransformerBlock(nn.Module):
    """
    A single block of a Vision Transformer. This includes a `MultiHeadAttention` layer and an `MLP`
    layer, with `LayerNorm`s before each layer.
    """
    
    def __init__(self, d_model: int, n_heads: int, r_mlp: int = 4, qkv_bias: bool = True):
        """
        Creates a new `nn.Module` that performs Transformer encoding with a pre-norm setup and
        `MultiHeadAttention` and `MLP` sublayers.
        
        Args:
            d_model (int): The dimensionality of the model
            n_heads (int): The number of heads to create in the `MultiHeadAttention` block
            r_mlp (int): How many dimensions the `MLP` should multiply by. Default = 4
            qkv_bias (bool): If true, the `qkv` projection will have a bias term
        """
        
        super().__init__()
        
        # store parameters
        self.d_model = d_model
        self.n_heads = n_heads
        
        # sublayer 1: Multi-Head attention
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, qkv_bias=qkv_bias)
        
        # sublayer 2: Multi-Layer Perceptron
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model),
        )
    
    def forward(self, x: torch.Tensor):
        # residual connection after sublayer 1
        x = x + self.mha(self.ln1(x))
        
        # residual connection after sublayer 2
        x = x + self.mlp(self.ln2(x))
        
        return x

class VisionTransformer(nn.Module):
    """
    A full implementation of a Vision Transformer with multiple `TransformerBlock`s. Handles patch embedding,
    positional encoding, and transformer blocks.
    """
    
    def __init__(self, d_model: int, patch_size: int, n_heads: int,
            n_layers: int, d_head: int, n_channels: int = 3, img_size: int = 224,
            qkv_bias: bool = True, training: bool = False):
        """
        Creates a new `nn.Module` that connects a few different components in order to implement a full
        Vision Transformer.
        
        Args:
            d_model (int): The dimensionality of the model
            img_size (int): The standard image size accepted by this model
            patch_size (int): The patch size used for patch embeddings
            n_channels (int): The number of image channels. 1 for greyscale, 3 for RGB, etc.
            n_heads (int): The number of heads to create in the `MultiHeadAttention` block 
            n_layers (int): The number of `TransformerBlock`s to create
            qkv_bias (bool): If `true`, the `qkv` projection will have a bias term
            training (bool): If `true`, this model will be considered in "training" mode. This primarily
                adjusts what the `forward` function returns. During training, this is the DINO projection
                head. Otherwise, this is just the list of tokens
        """
        
        super().__init__()
        
        # safety checks
        assert img_size % patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # store parameters
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.training = training
        
        # create a blank classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.tokens: torch.Tensor | list[torch.Tensor] = None
        
        # create encoders
        self.patch_embedding = PatchEmbeddings(d_model, patch_size, n_channels)
        self.positional_encoding = LearnedPositionalEncoding(d_model, img_size, patch_size)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, qkv_bias) for _ in range(n_layers)])
        
        self.norm = nn.LayerNorm(d_model)
        
        # create heads
        self.proj_head = DINOHead(d_model, d_head)
        
        # initialize parameters
        nn.init.trunc_normal_(self.cls_token, std=.02)
    
    def process_images(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        
        # embded each patch
        x = self.patch_embedding(images)
        
        # add class token
        tokens_batch = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((tokens_batch, x), dim=1)
        
        # add positional encoding
        x = self.positional_encoding(x, W, H)
        
        # run through transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x
    
    def forward(self, images: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(images, list):
            # count number of unique resolutions
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([img.shape[-1] for img in images]),
                return_counts=True,
            )[1], 0)
            
            # iterate through each batch
            start_idx = 0
            output: torch.Tensor = torch.empty(0).to(images[0].device)
            self.tokens = []
            
            for end_idx in idx_crops:
                out = self.process_images(torch.cat(images[start_idx:end_idx]))
                self.tokens.append(out)
                
                output = torch.cat((output, out[:, 0]))
                start_idx = end_idx
            
            # collapse single-resolution token lists
            if len(self.tokens) == 1:
                self.tokens = self.tokens[0]
            
            if self.training:
                return self.proj_head(output)
            else:
                return self.tokens
        else:
            self.tokens = self.process_images(images)
            
            if self.training:
                return self.proj_head(self.tokens)
            else:
                return self.tokens

    def get_class_token(self) -> torch.Tensor | list[torch.Tensor]:
        """
        Returns the most recent batch of class tokens (if `forward` was called with
        multiple image resolutions, this will be a list of batches)
        """
        
        if self.tokens is not None:
            if isinstance(self.tokens, list):
                return [token[:, 0] for token in self.tokens]
            else:
                return self.tokens[:, 0]

        return None

    def get_patch_tokens(self) -> torch.Tensor | list[torch.Tensor]:
        """
        Returns the most recent patch_tokens
        """
        
        if self.tokens is not None:
            if isinstance(self.tokens, list):
                return [token[:, 1:] for token in self.tokens]
            else:
                return self.tokens[:, 1:]

        return None