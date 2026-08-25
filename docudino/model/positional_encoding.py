import math

import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, img_size: int, patch_size: int):
        """
        Creates a new `nn.Module` that appends a class token and positional encoding to a
        sequence of patches.
        
        Args:
            d_model (int): The dimensionality of the model
            img_size (int): The size of the image to be processed
            patch_size (int): The size of the patches taken from the image
        """
        
        super().__init__()
        
        # store parameters
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        
        # create a blank positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, d_model))
        
        # initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x: torch.Tensor, w: int, h: int):
        n_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if n_patches == N and w == h:
            return x + self.pos_embed
        
        # interpolate position embeds | x: (batches, patches, dimensions)
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        
        # apparently this fixes something with floating point precision
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        
        w0, h0 = w0 + 0.1, h0 + 0.1
        
        patch_pos_embed = nn.functional.interpolate(
            # (batch, patch, dim) -> (batch, w, h, dim) -> (batch, dim, w, h)
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic'
        )
        
        # (batch, dim, w, h) -> (batch, w, h, dim) -> (batch, patch, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return x + torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)