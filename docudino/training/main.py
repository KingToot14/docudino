import sys
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from .util import get_params_groups
from .dino_loss import DINOLoss
from docudino.model import dino_v1, VisionTransformer
from docudino.data import DocumentDataset, DocumentSampler, TrainingAugmentations

class TrainingSystem:
    def __init__(self, config_file: str):
        # TODO: parse config file
        pass

        # set torch flags
        torch.backends.cudnn.benchmark = True
        # torch.backends.fp32_precision = "tf32"

        # create dataset
        transform = TrainingAugmentations(
            (0.40, 1.00),
            (0.10, 0.40),
            8,
        )
        
        self.dataset = DocumentDataset("datasets/historical_wi/train", 256, 256, transform=transform)
            
        self.dataloader = DataLoader(
            self.dataset,
            sampler=DocumentSampler(self.dataset),
            batch_size=128,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=3,
            pin_memory=True,
        )
        
        self.EPOCHS = 30
        
        # load model
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.student = dino_v1.vit_small(d_head=8192, training=True).to(self.DEVICE)
        self.teacher = dino_v1.vit_small(d_head=8192, training=True).to(self.DEVICE)
        
        self.student.compile(mode="max-autotune", backend="inductor")
        self.teacher.compile(mode="max-autotune", backend="inductor")
        
        # loss and optimizer
        self.criterion = DINOLoss(self.student.d_head, 10, 0.07, 0.04, 10, 0.10, self.EPOCHS, 0.90).to(self.DEVICE)
        self.optimizer = optim.AdamW(get_params_groups(self.student))
    
    def train(self):
        for epoch in range(self.EPOCHS):
            self.train_epoch(epoch)
        
        print("Done training")
    
    def train_epoch(self, epoch: int):
        print(f"Training ({epoch}/{self.EPOCHS})")
        
        running_loss = 0.0
        
        for images in tqdm(self.dataloader):
            # TODO: update weight decay and lr schedule
            pass
        
            # move images to gpu
            images: list[torch.Tensor] = [img.to(self.DEVICE, non_blocking=True) for img in images]
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                g_views = torch.cat(images[:2])
                l_views = torch.cat(images[2:])
                
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
            self.optimizer.step()
            
            # EMA teacher update
            with torch.no_grad():
                # TODO: load momentum
                momentum = 0.996
                
                for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                    param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)

            running_loss += loss.item()
        
        print(f"Loss: {running_loss / len(self.dataloader)}")

if __name__ == "__main__":
    training = TrainingSystem("")
    
    training.train()