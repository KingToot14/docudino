import math
import random
from pathlib import Path

from bisect import bisect_right

from PIL import Image

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision.io import decode_image
import torchvision.transforms.v2 as T

from .util import pad_image, split_image
from .augmentations import TrainingAugmentations
from docudino.training.config import DocuDINOTrainingConfig
from docudino.evaluation.config import DocuDINOEvaluationConfig

# --- Constants --- #
EXTENSIONS = [".png", ".jpg", ".jpeg"]

# --- Functions --- #
TO_FLOAT = T.ToDtype(torch.float32, scale=True)
NORMALIZE = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

def standard_transform(resize_size: int = 224):
    """
    Defines a standard DINO-style transformation that converts an image to a tensor and
    resizes it
    """
    
    return T.Compose([
        T.ToImage(),
        T.Resize((resize_size, resize_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])

# --- Classes --- #
class DocumentDataset(Dataset):
    """
    A `Dataset` that loads and caches document files, splits them into windows, and returns 1
    window at a time. This is optimized for reading all windows of a document at once before
    moving on to the next document.
    """
    
    def __init__(self, root_dir: str | Path, window_size: int, stride: int,
                 return_idx: bool = False, transform=standard_transform(224)):
        """
        Args:
            root_dir (str | Path): The root directory to recursively load files from
            window_size (int): How large each window sample should be (square crop)
            stride (int): How much the pointer should move between samples
            return_idx (bool): If `true`, the dataset will return the document and
                writer indices alongside the window. This only works if the dataset
                names are in the form 'writer-document_id'
            transform: The transform to apply to each collected sample
        """
        
        self.window_size = window_size
        self.stride = stride
        self.return_idx = return_idx
        
        # store root dir
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root_dir = root_dir
        
        self.transform = transform
        
        # collect image locations
        self.cached_image: torch.Tensor = None
        self.cached_image_idx: int = 0
        self.images: list[tuple[Path, int, int]] = []
        self.image_info: list[tuple[int, int]] = []
        
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
            
            if self.return_idx:
                writer, doc_id = path.stem.split('-')
                self.image_info.append((int(writer), int(doc_id)))
            
            patch_idx += patch_w * patch_h
        
        self.patch_count = patch_idx
        
        self.patch_starts = [
            patch_start for _, _, patch_start in self.images
        ]

    def _load_image(self, path: str, image_idx: int) -> None:
        img = decode_image(path)
                    
        # pad the image, then pre-split it into windows (bulk processing should be cheaper)
        self.cached_image = split_image(
            pad_image(img, self.window_size, self.stride),
            self.window_size,
            self.stride
        )
        
        self.cached_image_idx = image_idx
    
    def __len__(self):
        return self.patch_count

    def __getitem__(self, index: int) -> torch.Tensor:
        image_idx = bisect_right(self.patch_starts, index) - 1
        path, _, patch_start = self.images[image_idx]
        
        if self.cached_image is None or self.cached_image_idx != image_idx:
            self._load_image(path, image_idx)
        
        patch_idx = index - patch_start
        
        result = self.cached_image[patch_idx]
        
        if self.transform is not None:
            result = self.transform(result)
        
        if self.return_idx:
            return result, *self.image_info[image_idx]
        
        return result
    
    def __getitems__(self, indices: list[int]) -> list[torch.Tensor]:
        results = []
        image_idx = 0
        
        # print(indices)
        
        for index in indices:
            path, patch_count, patch_start = self.images[image_idx]
            
            # check if a new document started (indices should already be grouped)
            if index >= patch_start + patch_count or index < patch_start:
                image_idx = bisect_right(self.patch_starts, index) - 1
                
                path, patch_count, patch_start = self.images[image_idx]
                
                # load new image
                if self.cached_image is None or self.cached_image_idx != image_idx:
                    self._load_image(path, image_idx)
            
            patch_idx = index - patch_start
            
            result = self.cached_image[patch_idx]
            
            if self.transform is not None:
                result = self.transform(result)
            
            # check for index information
            if self.return_idx:
                result = [result, *self.image_info[image_idx]]
            
            results.append(result)
        
        return results

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
                 rank: int = 0, num_replicas: int = 1, seed: int | None = None):
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
        
        # update seed
        if seed is None:
            self.seed = random.randint(0, 2**16 - 1)
        else:
            self.seed = seed
        
        self.epoch = 0
        self.assigned_documents = None
        self.target_patches = 0
        
        self.set_epoch(0)
        
        self._distribute_indices()
    
    def set_epoch(self, epoch: int) -> None:
        """
        Sets the current training epoch. This is used for deterministic seeding
        """
        
        self.epoch = epoch
    
    def __len__(self):
        return self.target_patches
    
    def __iter__(self):
        yielded_windows = 0
        
        # create deterministic generator
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed + self.epoch)
        
        # get assigned indices
        assigned = self.document_buckets[self.rank + self.epoch % self.num_replicas]
        
        document_indices: torch.Tensor
        
        if self.shuffle:
            document_indices = torch.randperm(len(assigned), generator=self.generator)
        else:
            document_indices = torch.arange(0, len(assigned))
        
        # iterate through each document
        for doc_index in document_indices:
            index = assigned[doc_index]
            _, patch_count, patch_start = self.data.images[index]
            
            # shuffle windows
            patch_indices: torch.Tensor
            
            if self.shuffle:
                patch_indices = torch.randperm(patch_count, generator=self.generator)
            else:
                patch_indices = torch.arange(0, patch_count)
            
            # yield each window
            for patch_idx in patch_indices:
                if yielded_windows >= self.target_patches:
                    return
                
                yield patch_idx + patch_start
                
                yielded_windows += 1
    
    def _distribute_indices(self) -> torch.Tensor:
        """
        Evenly distributes the shuffled document indices by their patch size using a greedy
        "Longest Processing Time" algorithm. This also guarantees that each dataloader will
        have the exact same number of batches, which is very important for DDP.
        """
        
        self.document_buckets = [[] for _ in range(self.num_replicas)]
        bucket_patches = [0] * self.num_replicas
        
        # greedily assign all documents to ranks
        indices = torch.arange(0, len(self.data.images))
        
        for index in sorted(indices, key=lambda i: self.data.images[i][1], reverse=True):
            _, patch_count, _ = self.data.images[index]
            
            # gets the minimum rank by their value in 'bucket_patches'
            rank = min(range(self.num_replicas), key=bucket_patches.__getitem__)
            
            self.document_buckets[rank].append(index)
            bucket_patches[rank] += patch_count
        
        # convert to tensor
        self.document_buckets = [torch.tensor(bucket, dtype=torch.int32) for bucket in self.document_buckets]
        
        # make sure each rank has the same number of batches
        bucket_patches = [patch / self.batch_size for patch in bucket_patches]
        
        min_batch = int(min(bucket_patches))
        self.target_patches = min_batch * self.batch_size

def window_collate(data: list[tuple]) -> torch.Tensor:
    """
    Collates a batch of data by grouping the window data, writer ids, and document ids into their own gropus
    
    Args:
        data (list[tuple]): The list of data in the form: (window, writer_id, document_id)
    """
    
    # split each document
    windows, writers, documents = zip(*data)
    
    return (
        torch.stack(windows),
        torch.tensor(writers, dtype=torch.int32),
        torch.tensor(documents, dtype=torch.int32),
    )

# --- Builders --- #
def create_training_dataloader(cfg: DocuDINOTrainingConfig, local_rank: int, world_size: int) -> DataLoader:
    """
    Builds the standard `TrainingAugmentations`, `DocumentDataset`, `DataLoader`, and `DistributedDataSampler`
    used in standard training, and connects them all together.
    
    Args:
        cfg (DocuDINOTrainingConfig): The config information for training
        local_rank (int): The local rank of this process
        world_size (int): The world size of the distributed training
    """
    
    transform = TrainingAugmentations(
        cfg.dataset.global_view_scale,
        cfg.dataset.local_view_scale,
        cfg.dataset.local_views,
    )

    dataset = DocumentDataset(
        cfg.dataset.root,
        cfg.dataset.window_size, cfg.dataset.window_stride,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        sampler=DistributedDocumentSampler(
            dataset, cfg.dataset.batch_size,
            rank=local_rank, num_replicas=world_size
        ),
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        prefetch_factor=cfg.dataset.prefetch_factor if cfg.dataset.num_workers > 0 else None,
        pin_memory=True,
    )
    
    return dataloader

def create_evaluation_dataloader(cfg: DocuDINOEvaluationConfig, is_training: bool,
                                 local_rank: int, world_size: int) -> DataLoader:
    """
    Builds the standard `DocumentDataset`, `DataLoader`, and `DistributedDataSampler`
    used in standard training, and connects them all together.
    
    Args:
        cfg (DocuDINOTrainingConfig): The config information for training
        local_rank (int): The local rank of this process
        is_training (bool): If `True`, loads the dataset using the training parameters
        world_size (int): The world size of the distributed training
    """
    
    dataset = DocumentDataset(
        f"datasets/historical_wi/{("train" if is_training else "test")}",
        cfg.extract.window_size, cfg.extract.train_stride if is_training else cfg.extract.test_stride,
        return_idx=True
    )
    
    dataloader = DataLoader(
        dataset,
        sampler=DistributedDocumentSampler(
            dataset, cfg.dataset.batch_size,
            rank=local_rank, num_replicas=world_size
        ),
        collate_fn=window_collate,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        prefetch_factor=cfg.dataset.prefetch_factor if cfg.dataset.num_workers > 0 else None,
        pin_memory=True,
    )
    
    return dataloader