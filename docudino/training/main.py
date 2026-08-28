import os
import sys
import math
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

from tqdm import tqdm

from .util import get_params_groups, cosine_scheduler
from .config import load_config_file, DocuDINOConfig
from .dino_loss import DINOLoss
from docudino.model import dino_v1
from docudino.data import DocumentDataset, DistributedDocumentSampler, TrainingAugmentations

def setup_ddp() -> None:
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.accelerator.set_device_index(local_rank)
    
    # setup accelerator
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    
    dist.init_process_group(backend)
    
    return local_rank

class TrainingSystem:
    def __init__(self, config_file: str, overrides: list[str]):
        # parse config file
        self.cfg = load_config_file(config_file, overrides)

        # setup DDP
        self.LOCAL_RANK = setup_ddp()
        self.WORLD_SIZE = dist.get_world_size()

        # set torch flags
        torch.backends.cudnn.benchmark = True

        # create dataset
        transform = TrainingAugmentations(
            self.cfg.dataset.global_view_scale,
            self.cfg.dataset.local_view_scale,
            self.cfg.dataset.local_views,
        )
        
        self.dataset = DocumentDataset(
            "datasets/historical_wi/train",
            self.cfg.dataset.window_size, self.cfg.dataset.window_stride,
            transform=transform)
        
        self.dataloader = DataLoader(
            self.dataset,
            sampler=DistributedDocumentSampler(
                self.dataset, self.cfg.dataset.batch_size,
                rank=self.LOCAL_RANK, num_replicas=self.WORLD_SIZE
            ),
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            prefetch_factor=self.cfg.dataset.prefetch_factor,
            pin_memory=True,
        )
        
        self.EPOCHS = self.cfg.training.epochs
        
        # load model
        self.DEVICE = torch.device(f"cuda:{self.LOCAL_RANK}")
        self.student = dino_v1.vit_small(
            d_head=self.cfg.model.dino_head_dimensions,
            training=True,
        ).to(self.DEVICE)
        self.teacher = dino_v1.vit_small(
            d_head=self.cfg.model.dino_head_dimensions,
            training=True,
        ).to(self.DEVICE)
        
        # compile and set to DDP
        if self.cfg.model.compile_mode != 'none':
            self.student.compile(mode=self.cfg.model.compile_mode, backend=self.cfg.model.compile_backend) 
            self.teacher.compile(mode=self.cfg.model.compile_mode, backend=self.cfg.model.compile_backend) 
        
        self.student_no_ddp = self.student
        self.student = nn.parallel.DistributedDataParallel(self.student, device_ids=[self.LOCAL_RANK])
        
        self.teacher_no_ddp = self.teacher
        self.teacher = nn.parallel.DistributedDataParallel(self.teacher, device_ids=[self.LOCAL_RANK])
        
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        
        # loss and optimizer
        self.criterion = DINOLoss(
            self.student_no_ddp.d_head,
            2 + self.cfg.dataset.local_views,
            self.cfg.teacher.temp_end, self.cfg.teacher.temp_start, self.cfg.teacher.warmup_epochs,
            self.cfg.student.temp,
            self.EPOCHS, self.cfg.teacher.center_momentum
        ).to(self.DEVICE)
        
        self.optimizer = optim.AdamW(get_params_groups(self.student_no_ddp))
        
        # create schedulers
        self.weight_decay = cosine_scheduler(
            self.cfg.model.weight_decay_start, self.cfg.model.weight_decay_end,
            self.EPOCHS, len(self.dataloader),
        )
        
        # TODO: add config file
        self.learning_rate = cosine_scheduler(
            self.cfg.model.learning_rate * (self.cfg.dataset.batch_size * self.WORLD_SIZE / 256), 0.0,
            self.EPOCHS, len(self.dataloader),
            self.cfg.model.learning_rate_warmup_epochs, 0.0,
        )
        
        self.momentum = cosine_scheduler(
            self.cfg.teacher.ema_momentum_start, self.cfg.teacher.ema_momentum_end,
            self.EPOCHS, len(self.dataloader),
        )
    
    def train(self):
        print(f"Starting DINO training (Process {self.LOCAL_RANK})")
        
        if self.LOCAL_RANK == 0:
            print("NOTE: `torch.compile` is active, so first epoch may take a while (5+ minutes)")
        
        for epoch in range(self.EPOCHS):
            self.dataloader.sampler.set_epoch(epoch)
            self.train_epoch(epoch)
        
        print(f"Done training (Process {self.LOCAL_RANK})")
        
        dist.destroy_process_group()
    
    def train_epoch(self, epoch: int):
        if self.LOCAL_RANK == 0:
            print(f"Training ({epoch}/{self.EPOCHS})")
        
        running_loss = 0.0
        
        # only update progress on main thread
        data = self.dataloader
        if self.LOCAL_RANK == 0:
            data = tqdm(data)
        
        for it, images in enumerate(data):
            # update weight decay and lr schedule
            it = len(self.dataloader) * epoch + it
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.learning_rate[it]
                if i == 0:
                    param_group['weight_decay'] = self.weight_decay[it]
        
            # move images to gpu
            images: list[torch.Tensor] = [img.to(self.DEVICE, non_blocking=True) for img in images]
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # split views
                g_views = torch.cat(images[:2])
                l_views = torch.cat(images[2:])
                
                # mark start of iteration
                torch.compiler.cudagraph_mark_step_begin()
                
                # run models
                with torch.no_grad():
                    t_out: torch.Tensor = self.teacher(g_views)
                s_out: torch.Tensor = self.student([g_views, l_views])
                
                loss = self.criterion(s_out, t_out, epoch)
            
            loss_value = loss.item()
            
            if not math.isfinite(loss_value):
                print(f"Loss is non-finite: '{loss_value}', stopping training")
                sys.exit(1)
            
            # student update
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # clip gradient and freeze last layer
            torch.nn.utils.clip_grad_norm_(self.student_no_ddp.parameters(), 3.0)
            if epoch < self.cfg.model.frozen_epochs:
                for n, p in self.student_no_ddp.named_parameters():
                    if "last_layer" in n:
                        p.grad = None
            
            self.optimizer.step()
            
            # EMA teacher update
            with torch.no_grad():
                # load momentum
                momentum = self.momentum[it]
                
                for param_s, param_t in zip(self.student_no_ddp.parameters(), self.teacher.parameters()):
                    param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)

            running_loss += loss_value
        
        loss_sum = torch.tensor(running_loss, device=self.DEVICE)
        
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        
        if self.LOCAL_RANK == 0:
            print(f"Loss: {loss_sum / (len(self.dataloader) * self.WORLD_SIZE)}")

if __name__ == "__main__":
    # parse config file location
    parser = ArgumentParser(
        "DocuDINO Training",
        description="A DINO-style training system designed specifically for historical, handwritten documents",
    )
    
    parser.add_argument(
        "config", help="The location of the config file to load for training"
    )
    
    args, overrides = parser.parse_known_args()
    
    # create training system
    training = TrainingSystem(args.config, overrides)
    
    # start training
    training.train()