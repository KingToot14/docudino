import torch
import torch.nn.functional as F

def pad_image(image: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Pads the `image` to a multiple of `stride`. This creates a new copy of the tensor
    """

    # get size
    h, w = image.size()[-2:]

    pad_w = (-w) % stride
    pad_h = (-h) % stride

    return F.pad(image, [0, pad_w, 0, pad_h])

def split_image(image: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """
    Returns a view of image` that's been split into a collection of windows of
    `window_size`x`window_size`
    
    Args:
        image (torch.Tensor): The image to be split into windows
        window_size (int): How large the square windows should be
        stride (int): The distance between the top-left corner of each window
    """
    
    # This does a few things all at once:
    #  - Creates a new view using unfold, which creates additional dimensions to show where data is located
    #    this creates a tensor of shape: [channels, rows, cols, height, width]
    #  - Since we want the shape to be [windows, channels, height, width], we need to rearrange the dimensions
    #    and combine the rows and columns (permutate, then reshape)
    
    return image\
        .unfold(1, window_size, stride)\
        .unfold(2, window_size, stride)\
        .permute(1, 2, 0, 3, 4)\
        .reshape(-1, image.shape[0], window_size, window_size)
