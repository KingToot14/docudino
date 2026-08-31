from argparse import ArgumentParser

from tqdm import tqdm

from writer_retrieval.data import load_documents
from writer_retrieval.models import VLADCodebook, PCAMatrix
from writer_retrieval.config import Config

import torch

def train_vald_and_pca(config: Config) -> None:
    """
    Trains the VLAD codebook and the PCA whitening matrix on the patches found in `root/train`
    """
    
    name = config.run_name
    
    documents = load_documents(f"output/patches/{name}/train")
    target_samples = config.training.samples_per_document
    
    # collect subset for VLAD
    patches = []
    for document in documents:
        patch_count = document.shape[0]
        
        # collect random sample (if more than target samples)
        if patch_count <= target_samples or target_samples == -1:
            patches.append(document)
        else:
            patches.append(document[torch.randperm(patch_count)[:target_samples]])
        
    # train VLAD
    codebook = VLADCodebook()
    codebook.train(torch.cat(patches), k=config.training.codebook_clusters, niter=config.training.codebook_iterations)
    
    # save VLAD
    codebook.save(f"output/models/{name}/vlad.pt")
    
    # create document descriptors
    descriptors = []
    
    for document in tqdm(documents, desc="Creating VLAD Descriptors"):
        descriptors.append(codebook.create_descriptor(document))
    
    descriptors = torch.stack(descriptors)
    
    # train PCA
    pca_model = PCAMatrix()
    pca_model.train(descriptors, target_dims=config.training.pca_dimensions)
    
    # save PCA
    pca_model.save(f"output/models/{name}/pca.model")

if __name__ == "__main__":
    # create parser
    parser = ArgumentParser()
    parser.add_argument("config_file")
    
    # parse arguments
    args = parser.parse_args()
    
    config: Config = Config.from_yaml(args.config_file)
    
    train_vald_and_pca(config)