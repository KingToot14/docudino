from pathlib import Path

import torch
import torch.nn.functional as F

import faiss
import faiss.contrib.torch_utils

class VLADCodebook():
    def train(self, patches: torch.Tensor, k: int = 100, niter: int = 25) -> None:
        """
        Trains a VLAD codebook using FAISS's KMeans
        
        Args:
            patches (torch.Tensor): the patches to be used in training. This should be a subset of the full patches
            centroids (int): the number of centroids to use for the codebook (100 recommended)
            niter (int): the number of training iterations to perform
        """
        
        self.k: int = k
        self.dimensions: int = patches.shape[1]
        
        # create kmeans
        self.kmeans = faiss.Kmeans(
            d=self.dimensions,
            k=k,
            niter=niter,
            verbose=True,
            gpu=True,
        )
        
        # trains kmeans
        self.kmeans.train(patches.cpu())
        self.centroids = torch.as_torch.tensor(
            self.kmeans.centroids,
            device=patches.device,
            dtype=patches.dtype
        )
        
        # set index
        self.res = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexFlatL2(
            self.res, self.dimensions
        )
        self.index.add(self.centroids)

    def create_descriptor(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Creates an aggregated document descriptor based on the document foreground `patches` by running
        VLAD aggregation through the trained codebook
        """
        
        # calculate closest centroids
        _, indices = self.index.search(patches, 1)
        indices = indices.squeeze()
        residuals = patches - self.centroids[indices]
        
        # build descriptor
        descriptor = torch.zeros(self.k, patches.shape[1], device=patches.device)
        
        descriptor.index_add_(0, indices, residuals)
        descriptor = descriptor.reshape(-1)
        
        # power normalization
        descriptor = torch.sign(descriptor) * torch.abs(descriptor).pow(0.5)
        
        # l2 normalization
        descriptor = F.normalize(descriptor, p=2, dim=0)
        
        return descriptor
    
    def save(self, path: str) -> None:
        """
        Saves the trained VLAD clusters for later use in a new file at `path`
        """
        Path("/".join(path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "centroids": self.centroids,
            "k": self.k,
            "dimensions": self.dimensions,
        }, path)
    
    def load(self, path: str) -> None:
        """
        Loads the VLAD clusters and creates a new index from the file at `path`
        """
        
        # load centroids
        data = torch.load(path)
        self.centroids = data["centroids"]
        self.k = data["k"]
        self.dimensions = data["dimensions"]
        
        # create index
        self.res = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexFlatL2(
            self.res, self.dimensions
        )
        self.index.add(self.centroids)