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

from .util import get_params_groups, cosine_scheduler, load_config_file
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
    def __init__(self, config_file: str):
        # parse config file
        self.cfg = load_config_file(config_file)
    
        self.freeze_layer = 1

        # setup DDP
        self.LOCAL_RANK = setup_ddp()
        self.WORLD_SIZE = dist.get_world_size()

        # set torch flags
        torch.backends.cudnn.benchmark = True

        # create dataset
        transform = TrainingAugmentations(
            (0.40, 1.00),
            (0.10, 0.40),
            8,
        )
        
        self.dataset = DocumentDataset("datasets/historical_wi/train", 256, 256, transform=transform)
        
        self.dataloader = DataLoader(
            self.dataset,
            sampler=DistributedDocumentSampler(
                self.dataset, 128,
                rank=self.LOCAL_RANK, num_replicas=self.WORLD_SIZE
            ),
            batch_size=128,
            num_workers=4,
            prefetch_factor=3,
            pin_memory=True,
            drop_last=True,
        )
        
        self.EPOCHS = 30
        
        # load model
        self.DEVICE = torch.device(f"cuda:{self.LOCAL_RANK}")
        self.student = dino_v1.vit_small(d_head=8192, training=True).to(self.DEVICE)
        self.teacher = dino_v1.vit_small(d_head=8192, training=True).to(self.DEVICE)
        
        # compile and set to DDP
        self.student.compile(mode="max-autotune-no-cudagraphs", backend="inductor")
        self.teacher.compile(mode="max-autotune-no-cudagraphs", backend="inductor")
        
        self.student_no_ddp = self.student
        self.student = nn.parallel.DistributedDataParallel(self.student, device_ids=[self.LOCAL_RANK])
        
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        
        # loss and optimizer
        self.criterion = DINOLoss(self.student_no_ddp.d_head, 10, 0.07, 0.04, 10, 0.10, self.EPOCHS, 0.90).to(self.DEVICE)
        self.optimizer = optim.AdamW(get_params_groups(self.student_no_ddp))
        
        # create schedulers
        self.weight_decay = cosine_scheduler(
            0.04, 0.4,
            self.EPOCHS, len(self.dataloader),
        )
        
        # TODO: add config file
        self.learning_rate = cosine_scheduler(
            5e-4 * (128 / 256), 0.0,
            self.EPOCHS, len(self.dataloader),
            10, 0.0,
        )
        
        self.momentum = cosine_scheduler(
            0.996, 1.0,
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
            
            if not math.isfinite(loss.item()):
                print(f"Loss is non-finite: '{loss.item()}', stopping training")
                sys.exit(1)
            
            # student update
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # clip gradient and freeze last layer
            torch.nn.utils.clip_grad_norm_(self.student_no_ddp.parameters(), 3.0)
            if epoch < self.freeze_layer:
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

            running_loss += loss.item()
        
        loss_sum = torch.tensor(running_loss, device=self.DEVICE)
        
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        
        if self.LOCAL_RANK == 0:
            print(f"Loss: {running_loss / len(self.dataloader)}")

if __name__ == "__main__":
    # parse config file location
    parser = ArgumentParser(
        "DocuDINO Training",
        description="A DINO-style training system designed specifically for historical, handwritten documents",
    )
    
    parser.add_argument(
        "config", help="The location of the config file to load for training"
    )
    
    args = parser.parse_args()
    
    # create training system
    training = TrainingSystem(args.config)
    
    # start training
    training.train()