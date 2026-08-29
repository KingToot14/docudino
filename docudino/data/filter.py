from torch import Tensor
from torch.nn.functional import avg_pool2d

def get_window_filter(windows: Tensor, threshold: float = 0.025) -> Tensor:
    """
    Returns a mask that can be used to filter out the set of `windows` that contain
    less than `threshold` percentage of foreground pixels
    
    Args:
        windows (Tensor): the windows to filter
        threshold (float): the proportion of foreground pixels each window much have
            in order to be kept
    
    Returns:
        window_mask (Tensor): the mask to be applied to the window tensors
    """
    
    return windows.mean(dim=(1, 2, 3)) >= threshold

def get_patch_filter(windows: Tensor, threshold: int = 10) -> Tensor:
    """
    Returns a mask that can be used to filter out the set of patches that contain
    less than 'threshold' number of pixels from each of the `windows`
    
    Args:
        windows (Tensor): the windows to filter
        threshold (int): the number of foreground pixels each patch much have
            in order to be kept
    
    Returns:
        patch_mask (Tensor): the mask to be applied to the patch tensors
    """
    
    mean = avg_pool2d(windows.mean(dim=1), 16, 16).flatten(start_dim=1) * 256
    
    return mean >= threshold