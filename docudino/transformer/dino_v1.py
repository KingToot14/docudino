from .vit import VisionTransformer

def vit_tiny(patch_size: int = 16, **kwargs) -> VisionTransformer:
    """
    Constructs a tiny `VisionTransformer` according to DINOv1's implementation:
      - Embedding Dimensions (`d_model`) = 192
      - Depth (`n_layers`): 12
      - Heads (`n_heads`): 3
    """
    
    return VisionTransformer(
        d_model=192, patch_size=patch_size,
        n_layers=12, n_heads=3, qkv_bias=True,
        **kwargs,
    )

def vit_small(patch_size: int = 16, **kwargs) -> VisionTransformer:
    """
    Constructs a small `VisionTransformer` according to DINOv1's implementation:
        - Embedding Dimensions (`d_model`) = 384
        - Depth (`n_layers`): 12
        - Heads (`n_heads`): 6
    """
    
    return VisionTransformer(
        d_model=384, patch_size=patch_size,
        n_layers=12, n_heads=6, qkv_bias=True,
        **kwargs,
    )

def vit_base(patch_size: int = 16, **kwargs) -> VisionTransformer:
    """
    Constructs a base `VisionTransformer` according to DINOv1's implementation:
        - Embedding Dimensions (`d_model`) = 768
        - Depth (`n_layers`): 12
        - Heads (`n_heads`): 12
    """
    
    return VisionTransformer(
        d_model=768, patch_size=patch_size,
        n_layers=12, n_heads=12, qkv_bias=True,
        **kwargs,
    )