from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import torch.nn as nn

import numpy as np

def get_params_groups(model: nn.Module):
    """
    Groups the model's parameters by whether or not they should be weight regularized. Parameters
    are regularized when they are not a bias and when they are multi-dimensional
    
    Args:
        model (nn.Module): The model to get the parameter groups from
    """
    
    # separate regularized parameters
    regularized = []
    not_regularized = []
    
    # collect parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(start_value: float, end_value: float, epochs: int, iters_per_epoch: int,
                     warmup_epochs: int = 0, warmup_start_value: float = 0.0) -> np.ndarray:
    """
    Creates a cosine schedule that interpolates from `start_value` to `end_value` with an
    optional warmup time.
    
    Args:
        start_value (float): The value at the start of the schedule. If there is a warmup, this
            is the value at the end of the warmup
        end_value (float): The value at the end of the schedule
        epochs (int): The total number of epochs to schedule for
        iters_per_epoch (int): The number of iterations for each epoch. This is typically the
            length of the `DataLoader`
        warmup_epochs (int): How many epochs to warm up for. Defaults to 0
        warmup_start_value (float); The value at the start of the schedule if there is a warmup
            phase. Defaults to 0.0
    """
    
    # create warmup schedule
    warmup_schedule: np.ndarray = np.array([])
    warmup_iters: int = warmup_epochs * iters_per_epoch
    
    if warmup_iters > 0:
        warmup_schedule = np.linspace(warmup_start_value, start_value, warmup_iters)
    
    # create remaining schedule
    iters = np.arange(epochs * iters_per_epoch - warmup_iters)
    schedule = end_value + 0.5 * (start_value - end_value) * (1.0 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concat([warmup_schedule, schedule])
    
    assert len(schedule) == epochs * iters_per_epoch, "Schedule does not match the expected number of iterations"
    
    return schedule

def load_config_file(file: str | Path) -> DictConfig:
    """
    Parses the YAML config file localed at `file`, and overwrites from command-line arguments
    
    Args:
        file (str | Path): The config file's location
    
    Returns:
        DictConfig: the merged config object
    """
    
    config = OmegaConf.load(file)
    cli_config = OmegaConf.from_cli()
    
    return OmegaConf.merge(config, cli_config)