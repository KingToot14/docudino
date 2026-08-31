from pathlib import Path
from argparse import ArgumentParser

from writer_retrieval.data import load_patch
from writer_retrieval.models import VLADCodebook, PCAMatrix
from writer_retrieval.retrieval import WriterIndex, Metrics
from writer_retrieval.config import Config

import torch
from torch import Tensor
import torch.nn.functional as F

from tqdm import tqdm

def retrieve_writers(config: Config) -> None:
    """
    Loads all the testing data in `root/test` and passes it into a FAISS index for top-k and mAP metrics
    """
    
    name = config.run_name
    
    paths = sorted(Path(f"output/patches/{name}/test").rglob("*"))
    descriptors: list[Tensor] = []
    writers: list[int] = []
    
    # load models
    codebook = VLADCodebook()
    codebook.load(f"output/models/{name}/vlad.pt")
    
    pca_model = PCAMatrix()
    pca_model.load(f"output/models/{name}/pca.model")
    
    # load documents in batches
    for path in tqdm(paths, desc="Creating VLAD Descriptors"):
        for document, writer, doc_id in load_patch(path):
            # create VLAD descriptor
            descriptor = codebook.create_descriptor(document)
            
            # apply PCA whitening
            descriptor = torch.as_tensor(pca_model.apply(descriptor.unsqueeze(0)))
            descriptor = F.normalize(descriptor, p=2, dim=1)
            
            # add to list
            descriptors.append(descriptor)
            writers.append(writer)
    
    # collect descriptors
    descriptors: Tensor = torch.cat(descriptors)
    writers: Tensor = torch.as_tensor(writers)
    
    # create index
    index = WriterIndex(descriptors.shape[-1])
    index.add(descriptors)
    
    # calculate metrics
    met = Metrics(descriptors, writers, index)
    
    # calculate metrics
    met.run_metrics(
        "output/metrics/results.csv",
        name,
        config.evaluation.metrics
    )

if __name__ == "__main__":
    # create parser
    parser = ArgumentParser()
    parser.add_argument("config_file")
    
    # parse arguments
    args = parser.parse_args()
    
    config: Config = Config.from_yaml(args.config_file)
    
    retrieve_writers(config)