from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from typing import List
from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    """
    Stores config info for the training dataset
    """
    
    root: str = ""
    """The root filepath of the training dataset"""
    window_size: int = 256
    """The size of the windows to slice from each image in the dataset"""
    window_stride: int = 256
    """The distance to move when slicing multiple windows from an image. A value equal
    to `window_size` means it slices a perfectly non-overlapping collection of windows"""
    global_view_scale: List[float] = field(default_factory=lambda: [0.4, 1.0])
    """How much area each global view should take from the base image. Should be a 2-length
    list with the minimum and maximum size"""
    local_view_scale: List[float] = field(default_factory=lambda: [0.1, 0.4])
    """How much area each local view should take from the base image. Should be a 2-length
    list with the minimum and maximum size"""
    local_views: int = 8
    """How many local views to take from the """
    batch_size: int = 128
    """How many windows should be included in a single batch. Increases throughput and memory usage"""
    num_workers: int = 4
    """How many CPU workers to create when loading the dataset"""

@dataclass
class DocuDINOConfig:
    """
    Stores config info for all of DocuDINO
    """
    
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    """The dataset-specific config info"""

def load_config_file(file: str | Path, overrides: list[str]) -> DocuDINOConfig:
    """
    Parses the YAML config file localed at `file`, and overwrites from command-line arguments
    
    Args:
        file (str | Path): The config file's location
        overrides (list[str]): The remaining command-line arguments to be used as config overrides
    
    Returns:
        DictConfig: the merged config object
    """
    
    # get structure info
    schema = OmegaConf.structured(DocuDINOConfig)
    
    # load config file from path
    config = OmegaConf.load(file)
    cli_config = OmegaConf.from_dotlist(overrides)
    
    return OmegaConf.merge(schema, config, cli_config)