import os
from argparse import ArgumentParser

from time import perf_counter

import torch
import torch.distributed as dist

from tqdm import tqdm

from docudino.evaluation.config import load_evaluation_config
from docudino.data import create_evaluation_dataloader, TO_FLOAT, NORMALIZE
from docudino.data.filter import get_window_filter, get_patch_filter
from docudino.data.serialization import save_patches, AsyncWriter
from docudino.model import dino_v1

def setup_ddp() -> None:
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.accelerator.set_device_index(local_rank)
    
    # setup accelerator
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    
    dist.init_process_group(f"cpu:gloo,cuda:{backend}")
    
    return local_rank

def close_ddp() -> None:
    dist.destroy_process_group()

class EvaluationSystem:
    def __init__(self, config_file: str, run_name: str, overrides: list[str]):
        # parse config file
        self.cfg = load_evaluation_config(config_file, overrides)
        self.run_name = run_name

        # setup DDP
        self.LOCAL_RANK = setup_ddp()
        self.WORLD_SIZE = dist.get_world_size()
        self.DEVICE = torch.device(f"cuda:{self.LOCAL_RANK}")

        # set torch flags
        torch.backends.cudnn.benchmark = True
        
        # create model
        self.model = dino_v1.vit_small().to(self.DEVICE)
        self.model.eval()
        
        if self.cfg.extract.compile_mode != 'none':
            if self.LOCAL_RANK == 0:
                print("NOTE: `torch.compile` is active, so first epoch may take a while (5+ minutes)")
            
            self.model.compile(mode=self.cfg.extract.compile_mode, backend=self.cfg.extract.compile_backend)
    
    def extract_dataset(self, is_training: bool) -> None:
        """
        Extracts a collection of images using a Vision Transformer
        """
        
        self.dataloader = create_evaluation_dataloader(self.cfg, is_training, self.LOCAL_RANK, self.WORLD_SIZE)
        self.normalize = NORMALIZE.to(self.DEVICE, non_blocking=True)
        
        print(len(self.dataloader.dataset))
        
        # file saving
        writer = AsyncWriter()
        batch_tokens = []
        batch_writers = []
        batch_documents = []
        
        # start processing
        data = self.dataloader
        
        if self.LOCAL_RANK == 0:
            data = tqdm(data)
        
        for i, batch in enumerate(data):
            windows, writers, documents = batch
            
            # move to GPU and convert to float
            windows: torch.Tensor = windows.to(self.DEVICE, non_blocking=True)
            
            # get filters
            mask_win = get_window_filter(windows)
            mask_patch = get_patch_filter(windows)
            mask = mask_patch & mask_win[:, None]
            
            # extract tokens
            windows = self.normalize(windows)
            
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # keep only patch tokens
                tokens = self.model(windows)[:, 1:]
            
            # filter patches
            window_ids, patch_ids = mask.nonzero(as_tuple=True)
            window_ids_cpu = window_ids.cpu()
            
            tokens = tokens[window_ids, patch_ids]
            writers = writers[window_ids_cpu]
            documents = documents[window_ids_cpu]
            
            # store tokens
            output_root = f"output/patches/{self.run_name}/train" if is_training else f"output/patches/{self.run_name}/test"
            writer.save(f"{output_root}/rank_{self.LOCAL_RANK}-patch_{i}.pt", tokens, writers, documents)
        
        writer.close()
        

if __name__ == "__main__":
    # parse config file location
    parser = ArgumentParser(
        "DocuDINO Training",
        description="A DINO-style training system designed specifically for historical, handwritten documents",
    )
    
    parser.add_argument(
        "config", help="The location of the config file to load for training"
    )
    parser.add_argument(
        "run_name", help="The name of this extraction run used to avoid overwriting previous runs"
    )
    
    args, overrides = parser.parse_known_args()
    
    # create evaluation system
    evaluation = EvaluationSystem(args.config, args.run_name, overrides)
    
    # extract datasets
    # evaluation.extract_dataset(True)
    evaluation.extract_dataset(False)
    
    close_ddp()