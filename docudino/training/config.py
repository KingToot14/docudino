from pathlib import Path
from omegaconf import OmegaConf

from typing import List
from dataclasses import dataclass, field

# --- Training Config --- #
@dataclass
class TrainingDatasetConfig:
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
    prefetch_factor: int = 3
    """The `prefetch_factor` to use for the Data Loader"""

@dataclass
class ModelConfig:
    """
    Stores config info for general model configuration
    """
    frozen_epochs: int = 1
    """How many epochs the model should be frozen for"""
    dino_head_dimensions: int = 8192
    """How many dimensions the DINO projection head has"""
    weight_decay_start: float = 0.04
    """What value the weight decay should start at"""
    weight_decay_end: float = 0.40
    """What value the weight decay should end at"""
    learning_rate: float = 5e-4
    """The base value for the learning rate. This is scaled by `batch_size / 128`"""
    learning_rate_warmup_epochs: int = 10
    """How many epochs to warm up the learning rate for"""
    compile_mode: str = 'max-autotune-no-cudagraphs'
    """What mode to use for `torch.compile`. A value of 'none' disables compilation"""
    compile_backend: str = 'inductor'
    """What compiler backend to use for `torch.compile`"""

@dataclass
class StudentConfig:
    """
    Stores config info for the student model
    """
    
    temp: float = 0.10
    """The temperature of the student output. This determines how opinionated the student is"""

@dataclass
class TeacherConfig:
    """
    Stores config info for the teacher model
    """
    
    temp_start: float = 0.04
    """The starting temperature of the teacher model"""
    temp_end: float = 0.07
    """The ending temperature of the teacher model"""
    warmup_epochs: int = 10
    """How long it takes the teacher to warm up"""
    center_momentum: float = 0.90
    """The momentum of the teacher's centering. Higher values mean more of the original center is
    kept after each iteration"""
    ema_momentum_start: float = 0.996
    """The starting momentum of the teacher's EMA updates"""
    ema_momentum_end: float = 1.0
    """The ending momentum of the teacher's EMA updates"""

@dataclass
class TrainingConfig:
    """
    Stores config info about the training process
    """
    
    epochs: int = 30
    """How many epochs to train for"""
    
@dataclass
class LoggingConfig:
    """
    Stores config info about logging
    """
    
    wandb: bool = True
    """If `True`, Weights & Biases output will be used"""

@dataclass
class DocuDINOTrainingConfig:
    """
    Stores config info for DocuDINO's training pipeline
    """
    
    dataset: TrainingDatasetConfig = field(default_factory=lambda: TrainingDatasetConfig())
    """The dataset-specific config info"""
    
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    """The general model config info"""
    
    student: StudentConfig = field(default_factory=lambda: StudentConfig())
    """The teacher model's config info"""
    teacher: TeacherConfig = field(default_factory=lambda: TeacherConfig())
    """The student model's config info"""
    
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    """The training config info"""
    
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())
    """The logging config info"""

# --- Loading --- #
def load_training_config(file: str | Path, overrides: list[str]) -> DocuDINOTrainingConfig:
    """
    Parses the YAML config file localed at `file`, and overwrites from command-line arguments
    
    Args:
        file (str | Path): The config file's location
        overrides (list[str]): The remaining command-line arguments to be used as config overrides
    
    Returns:
        DictConfig: the merged config object
    """
    
    # get structure info
    schema = OmegaConf.structured(DocuDINOTrainingConfig)
    
    # load config file from path
    config = OmegaConf.load(file)
    cli_config = OmegaConf.from_dotlist(overrides)
    
    return OmegaConf.merge(schema, config, cli_config)