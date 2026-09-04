from pathlib import Path
from omegaconf import OmegaConf

from typing import List
from dataclasses import dataclass, field

# --- Evaluation Config --- #
@dataclass
class EvaluationDatasetConfig:
    """
    Stores config info for the evaluation dataset
    """
    
    root: str = ""
    """The root filepath of the evaluation dataset"""
    
    dataset: str = ""
    """The type of dataset to use. Possible options are 'standard' and 'zarr'"""
    
    batch_size: int = 128
    """How many windows should be included in a single batch. Increases throughput and memory usage"""
    num_workers: int = 4
    """How many CPU workers to create when loading the dataset"""
    prefetch_factor: int = 3
    """The `prefetch_factor` to use for the Data Loader"""
    shuffle: bool = True
    """If `True`, the dataset will be shuffled on a per-document and per-patch level"""

@dataclass
class ExtractConfig:
    """Stores config info about the feature extraction phase of the pipeline"""
    
    weights: str = ""
    """The path to the weights file to load into the Vision Transformer"""
    window_size: int = 224
    """The size of the windows to slice from each image in the dataset"""
    train_stride: int = 224
    """The stride used for splitting the train documents into windows. For paper accuracy, this should be `224`"""
    test_stride: int = 56
    """The stride used for splitting the test documents into windows. For paper accuracy, this should be `56`"""
    compile_mode: str = 'max-autotune-no-cudagraphs'
    """What mode to use for `torch.compile`. A value of 'none' disables compilation"""
    compile_backend: str = 'inductor'
    """What compiler backend to use for `torch.compile`"""

@dataclass
class CodebookConfig:
    """Stores config info about the VLAD and PCA training phase of the pipeline"""
    
    samples_per_document: int = 512
    """How many random samples to take from each document for VLAD codebook training. This should target roughly
    `100,000` to `1,000,000` samples across all documents"""
    codebook_iterations: int = 250
    """How many iterations the codebook training should run for. This uses `FAISS.KMeans` behind the scenes, so it
    has an early exit to save computation"""
    codebook_clusters: int = 100
    """How many clusters to train the codebook with. For paper accuracy, this should be `100`"""
    pca_dimensions: int = 384
    """How many dimensions the `FAISS.PCAMatrix` should reduce the original VLAD descriptors to"""

@dataclass
class EvaluationConfig:
    """Stores config info about the evaluation and metrics phase of the pipeline"""
    
    metrics: list[str] = field(default_factory=lambda: [])
    """A list of the metrics to calculate for evaluation. This should be a list in the format: `<metric>:<k>`
    where `metric` is among: `topk` and `mAP`; and where `k` should either be omitted (for an all-document
    query) or the maximum number of documents to search for"""

@dataclass
class DocuDINOEvaluationConfig:
    """
    Stores config info for DocuDINO's evaluation pipeline
    """
    
    dataset: EvaluationDatasetConfig = field(default_factory=lambda: EvaluationDatasetConfig())
    """The dataset-specific config info"""
    
    extract: ExtractConfig = field(default_factory=lambda: ExtractConfig())
    """The extraction phase config info"""
    codebook: CodebookConfig = field(default_factory=lambda: CodebookConfig())
    """The codebook training phase config info"""
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig())
    """The evaluation phase config info"""

# --- Loading --- #
def load_evaluation_config(file: str | Path, overrides: list[str]) -> DocuDINOEvaluationConfig:
    """
    Parses the YAML config file localed at `file`, and overwrites from command-line arguments
    
    Args:
        file (str | Path): The config file's location
        overrides (list[str]): The remaining command-line arguments to be used as config overrides
    
    Returns:
        DictConfig: the merged config object
    """
    
    # get structure info
    schema = OmegaConf.structured(DocuDINOEvaluationConfig)
    
    # load config file from path
    config = OmegaConf.load(file)
    cli_config = OmegaConf.from_dotlist(overrides)
    
    return OmegaConf.merge(schema, config, cli_config)