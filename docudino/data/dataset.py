import math
from pathlib import Path

from bisect import bisect_right

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

from .util import pad_image, split_image

# --- Constants --- #
EXTENSIONS = [".png", ".jpg", ".jpeg"]

# --- Classes --- #
class DocumentDataset(Dataset):
    def __init__(self, root_dir: str | Path, window_size: int, stride: int, transform=None):
        """
        Args:
            root_dir (str | Path): The root directory to recursively load files from
            window_size (int): How large each window sample should be (square crop)
            stride (int): How much the pointer should move between samples
        """
        
        self.window_size = window_size
        self.stride = stride
        
        # store root dir
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root_dir = root_dir
        
        self.transform = transform
        
        # collect image locations
        self.cached_image: torch.Tensor = None
        self.cached_image_idx: int = 0
        self.images: list[tuple[int, Path, int, int]] = []
        
        patch_idx: int = 0
        
        for path in sorted(Path(root_dir).rglob("*")):
            # make sure file extension is supported
            if path.suffix.lower() not in EXTENSIONS:
                continue
            
            # collect image size (should just lazily load metadata, not actual image)
            w, h = 0, 0
            with Image.open(path) as img:
                w, h = img.size
            
            # get patches
            patch_w = math.ceil((w - window_size) / stride) + 1
            patch_h = math.ceil((h - window_size) / stride) + 1
            
            self.images.append((path, patch_idx))
            
            patch_idx += patch_w * patch_h
        
        self.patch_count = patch_idx
        
        self.patch_starts = [
            patch_start for _, patch_start in self.images
        ]
    
    def __len__(self):
        return self.patch_count

    def __getitem__(self, index: int) -> torch.Tensor:
        image_idx = bisect_right(self.patch_starts, index) - 1
        path, patch_start = self.images[image_idx]
        
        if self.cached_image is None or self.cached_image_idx != image_idx:
            img = decode_image(path)
            
            # pad the image, then pre-split it into windows (bulk processing should be cheaper)
            self.cached_image = split_image(
                pad_image(img, self.stride),
                self.window_size,
                self.stride
            )
            
            self.cached_image_idx = image_idx
        
        patch_idx = index - patch_start
        
        result = self.cached_image[patch_idx]
        
        if self.transform is not None:
            result = self.transform(result)
        
        return result