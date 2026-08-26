import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

class DINOLoss(nn.Module):
    def __init__(self, d_head: int, n_crops: int, t_temp: float,
                 warmup_t_temp: float, warmup_t_epochs: int, s_temp: float,
                 epochs: int, center_momentum: float):
        """
        Args:
            d_head (int): The embedding size of the projection head
            n_crops (int): The number of local + global crops
            t_temp (float): The teacher's final temperature. This controls how sharp the
                teacher's output is. This is warmed up from `warmup_t_temp` over `warmup_t_epochs`
            warmup_t_temp (float): The teacher's starting temperature
            warmup_t_epochs (int): How many epochs it takes for the teacher's temperature to
                fully warm up
            s_temp (float): The student's temperature. This is not warmed up at all
            epochs (int): The total number of epochs training will run for
            center_momentum (float): How much the center will update over time. This prevents
                training from collapsing to a single dimension
        """
        
        super().__init__()
        
        # store parameters
        self.student_temp = s_temp
        self.teacher_temp = np.concatenate([
            np.linspace(warmup_t_temp, t_temp, warmup_t_epochs),
            np.ones(epochs - warmup_t_epochs) * t_temp
        ])
        
        self.n_crops = n_crops
        
        # create center
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.ones(1, d_head))
    
    def forward(self, s_out: torch.Tensor, t_out: torch.Tensor, epoch: int) -> float:
        # split model output
        s_views = s_out / self.student_temp
        s_views = s_views.chunk(self.n_crops)
        
        temp = self.teacher_temp[epoch]
        t_views = F.softmax((t_out - self.center) / temp, dim=-1)
        t_views = t_views.detach().chunk(2) # detach to prevent updating gradients
        
        # calculate loss
        total_loss = 0
        n_loss_terms = 0
        
        for t in range(len(t_views)):
            for s in range(len(s_views)):
                # skip if views are the same
                if t == s:
                    continue
                
                # calculate loss
                loss = torch.sum(-t_views[t] * F.log_softmax(s_views[s], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
            
        # average loss
        total_loss /= n_loss_terms
        self.update_center(t_out)
        
        return total_loss

    @torch.no_grad()
    def update_center(self, t_out: torch.Tensor) -> None:
        # sum all batches from all GPUs
        batch_center = torch.sum(t_out, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        
        # calculate average output
        batch_center /= (len(t_out) * dist.get_world_size())
        
        self.center = (self.center_momentum * self.center) + ((1.0 - self.center_momentum) * batch_center)