import torch.nn as nn

def get_params_groups(model: nn.Module):
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