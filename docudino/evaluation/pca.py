from pathlib import Path

import faiss
import faiss.contrib.torch_utils

import torch

class PCAMatrix:
    def train(self, descriptors: torch.Tensor, target_dims: int = 384) -> None:
        """
        Trains a PCA model with whitening enabled. This reduces the dimensions of `descriptors` into
        `target_dims`
        """
        
        # create PCA
        self.pca = faiss.PCAMatrix(
            descriptors.shape[-1],
            target_dims,
            -0.5,           # whitening
        )
        
        self.pca.train(descriptors.cpu())
    
    def apply(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Applies the trained PCA model to the given `descriptors` (must be batched)
        """
        
        return self.pca.apply_py(descriptors.cpu())
    
    def save(self, path: str) -> None:
        """
        Saves the PCA model in a file for later use at the given `path`
        """
        
        Path("/".join(path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        
        faiss.write_VectorTransform(self.pca, path)
    
    def load(self, path: str) -> None:
        """
        Loads the PCA model from a file at the given `path`
        """
        
        self.pca = faiss.read_VectorTransform(path)