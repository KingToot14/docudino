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
        self.register_buffer("center", torch.zeros(1, d_head))
    
    def forward(self, s_out: torch.Tensor, t_out: torch.Tensor, epoch: int, do_metrics: bool = False) -> tuple[float, dict]:
        # split model output
        s_views = s_out.float() / self.student_temp
        s_views = s_views.chunk(self.n_crops)
        
        temp = self.teacher_temp[epoch]
        t_centered = t_out.float() - self.center.float()
        t_views = F.softmax(t_centered / temp, dim=-1)
        t_views = t_views.detach().chunk(2) # detach to prevent updating gradients
        
        # calculate metrics
        if do_metrics:
            with torch.no_grad():
                all_t = torch.cat(t_views)
                
                entropy = -(all_t * torch.log(all_t.clamp_min(1e-8))).sum(dim=-1)
                max_prob = all_t.max(dim=-1).values
                
                # store metrics
                metrics = {
                    "teacher/entropy": self.ddp_mean(entropy.mean()).item(),
                    "teacher/entropy_std": self.ddp_mean(entropy.std()).item(),
                    "teacher/perplexity": self.ddp_mean(entropy.exp().mean()).item(),
                    "teacher/max_prob": self.ddp_mean(max_prob.mean()).item(),
                    "teacher/max_prob_95": self.ddp_mean(torch.quantile(max_prob, 0.95)).item(),
                    "teacher/logit_mean": self.ddp_mean(t_centered.mean()).item(),
                    "teacher/logit_std": self.ddp_mean(t_centered.std()).item(),
                    "teacher/center_mean": self.ddp_mean(self.center.mean()).item(),
                    "teacher/center_std": self.ddp_mean(self.center.std()).item(),
                    "teacher/center_norm": self.ddp_mean(self.center.norm()).item(),
                }
        
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
        
        if do_metrics:
            return total_loss, metrics
        else:
            return total_loss

    @torch.no_grad()
    def ddp_mean(self, value: torch.Tensor) -> torch.Tensor:
        """
        Average the `value` tensor along all DDP ranks
        """
        value = value.clone()
        
        dist.all_reduce(value)
        value /= dist.get_world_size()
        
        return value

    @torch.no_grad()
    def update_center(self, t_out: torch.Tensor) -> None:
        # sum all batches from all GPUs
        batch_center = torch.sum(t_out.float(), dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        
        # calculate average output
        batch_center /= (len(t_out) * dist.get_world_size())
        
        self.center = (self.center_momentum * self.center) + ((1.0 - self.center_momentum) * batch_center)