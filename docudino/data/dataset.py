import math
import random
from pathlib import Path

from bisect import bisect_right

from PIL import Image

import torch
from torch.utils.data import Dataset, Sampler
from torchvision.io import decode_image
import torchvision.transforms.v2 as T

from .util import pad_image, split_image

# --- Constants --- #
EXTENSIONS = [".png", ".jpg", ".jpeg"]

# --- Functions --- #
def standard_transform(resize_size: int = 224):
    """
    Defines a standard DINO-style transformation that converts an image to a tensor,
    resizes it, and normalizes it
    """
    
    return T.Compose([
        T.ToImage(),
        T.Resize((resize_size, resize_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

# --- Classes --- #
class DocumentDataset(Dataset):
    """
    A `Dataset` that loads and caches document files, splits them into windows, and returns 1
    window at a time. This is optimized for reading all windows of a document at once before
    moving on to the next document.
    """
    
    def __init__(self, root_dir: str | Path, window_size: int, stride: int, transform=standard_transform(224)):
        """
        Args:
            root_dir (str | Path): The root directory to recursively load files from
            window_size (int): How large each window sample should be (square crop)
            stride (int): How much the pointer should move between samples
            transform: The transform to apply to each collected sample
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
        
        self.image_count: int = 0
        self.patch_count: int = 0
        
        for path in sorted(Path(root_dir).rglob("*")):
            # make sure file extension is supported
            if path.suffix.lower() not in EXTENSIONS:
                continue
            
            # collect image size (should just lazily load metadata, not actual image)
            w, h = 0, 0
            with Image.open(path) as img:
                w, h = img.size
            
            # get patches
            patch_w = max(1, math.ceil((w - window_size) / stride) + 1)
            patch_h = max(1, math.ceil((h - window_size) / stride) + 1)
            patch_count = patch_w * patch_h
            
            self.images.append((path, patch_count, patch_idx))
            self.image_count += 1
            
            patch_idx += patch_w * patch_h
        
        self.patch_count = patch_idx
        
        self.patch_starts = [
            patch_start for _, _, patch_start in self.images
        ]
    
    def __len__(self):
        return self.patch_count

    def __getitem__(self, index: int) -> torch.Tensor:
        image_idx = bisect_right(self.patch_starts, index) - 1
        path, _, patch_start = self.images[image_idx]
        
        if self.cached_image is None or self.cached_image_idx != image_idx:
            img = decode_image(path)
            
            # pad the image, then pre-split it into windows (bulk processing should be cheaper)
            self.cached_image = split_image(
                pad_image(img, self.window_size, self.stride),
                self.window_size,
                self.stride
            )
            
            self.cached_image_idx = image_idx
        
        patch_idx = index - patch_start
        
        result = self.cached_image[patch_idx]
        
        if self.transform is not None:
            result = self.transform(result)
        
        return result

class DocumentSampler(Sampler):
    """
    A `Sampler` that's aware of the optimization requires of `DocumentDatabase` i.e. loading
    all windows of a document before moving on to another document
    """
    
    def __init__(self, data: DocumentDataset, shuffle: bool = True):
        """
        Args:
            data (DocumentDataset): The dataset to sample from
            shuffle (bool): If `true`, the document load order will be randomized between
                each epoch
        """
        self.data = data
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        indices: torch.Tensor
        
        if self.shuffle:
            indices = torch.randperm(self.data.image_count)
        else:
            indices = torch.arange(0, self.data.image_count)
        
        for index in indices:
            _, patch_count, patch_start = self.data.images[index]
            
            patch_indices: torch.Tensor
            
            if self.shuffle:
                patch_indices = torch.randperm(patch_count)
            else:
                patch_indices = torch.arange(0, patch_count)
            
            for patch_idx in patch_indices:
                yield patch_idx + patch_start
        

class DistributedDocumentSampler(Sampler):
    """
    A `Sampler` that's aware of the optimization requires of `DocumentDatabase` i.e. loading
    all windows of a document before moving on to another document. This specific version also
    handles distrbuted data across multiple GPU processes
    """
    
    def __init__(self, data: DocumentDataset, batch_size: int, shuffle: bool = True,
                 rank: int = 0, num_replicas: int = 1, seed: int = random.randint(0, 2**16 - 1)):
        """
        Args:
            data (DocumentDataset): The dataset to sample from
            shuffle (bool): If `true`, the document load order will be randomized between
                each epoch
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.rank = rank
        self.num_replicas = num_replicas
        
        self.seed = seed
        
        self.epoch = 0
        self.assigned_documents = None
        self.target_patches = 0
        
        self.set_epoch(0)
    
    def set_epoch(self, epoch: int) -> None:
        """
        Sets the current training epoch. This is used for deterministic seeding
        """
        
        self.epoch = epoch
        
        # create deterministic generator
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed + self.epoch)
        
        # shuffle documents
        indices: torch.Tensor
        
        if self.shuffle:
            indices = torch.randperm(self.data.image_count, generator=self.generator)
        else:
            indices = torch.arange(0, self.data.image_count)
        
        # only keep this GPU's indicies
        self._distribute_indices(indices)
    
    def __len__(self):
        return self.target_patches
    
    def __iter__(self):
        yielded_docs = 0
        
        # iterate through each document
        for index in self.assigned_documents:
            _, patch_count, patch_start = self.data.images[index]
            
            # shuffle windows
            patch_indices: torch.Tensor
            
            if self.shuffle:
                patch_indices = torch.randperm(patch_count, generator=self.generator)
            else:
                patch_indices = torch.arange(0, patch_count)
            
            # yield each window
            for patch_idx in patch_indices:
                if yielded_docs >= self.target_patches:
                    return
                
                yield patch_idx + patch_start
                
                yielded_docs += 1
    
    def _distribute_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Evenly distributes the shuffled document `indices` by their patch size using a greedy
        "Longest Processing Time" algorithm. This also guarantees that each dataloader will
        have the exact same number of batches, which is very important for DDP.
        
        Args:
            indices (torch.Tensor): The shuffled document indices
        """
        
        rank_documents = [[] for _ in range(self.num_replicas)]
        rank_patches = [0] * self.num_replicas
        
        # greedily assign all documents to ranks
        for index in sorted(indices, key=lambda i: self.data.images[i][1], reverse=True):
            _, patch_count, _ = self.data.images[index]
            
            # gets the minimum rank by their value in 'rank_patches'
            rank = min(range(self.num_replicas), key=rank_patches.__getitem__)
            
            rank_documents[rank].append(index)
            rank_patches[rank] += patch_count
        
        # make sure each rank has the same number of documents
        rank_patches = [patch / self.batch_size for patch in rank_patches]
        
        min_batch = int(min(rank_patches))
        self.target_patches = min_batch * self.batch_size
        
        self. assigned_documents = indices[torch.isin(
            indices, torch.tensor(rank_documents[self.rank]), assume_unique=True
        )]