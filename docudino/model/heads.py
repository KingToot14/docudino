import torch
import torch.nn as nn

class DINOHead(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_layers: int = 3,
                 d_hidden: int = 2048, d_bottleneck: int = 256,
                 norm_last_layer: bool = True):
        """
        Constructs the DINO projection head, which is responsible for learning the class
        token into a good training metric. The main idea is to keep the global representation
        of each view as close to each other as possible.
        """
        
        # architecture: if layers == 1: just return linear
        # if layers >= 2: build up layers from in -> hidden, hidden -> hidden (repeating), then hidden -> bottleneck
        
        super().__init__()
        
        assert n_layers >= 1, "Projection head must have 'n_layers' of 1 or higher"
        
        # build up learning layers
        if n_layers == 1:
            self.mlp = nn.Linear(d_model, d_bottleneck)
        else:
            layers: list[nn.Module] = [nn.Linear(d_model, d_hidden), nn.GELU()]
            
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(d_hidden, d_hidden))
                layers.append(nn.GELU())
            
            layers.append(nn.Linear(d_hidden, d_bottleneck))
            
            self.mlp = nn.Sequential(*layers)
        
        # projection layer
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(d_bottleneck, d_head, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def _init_weights(self, m: nn.Module) -> None:
        """
        Initializes the MLP's linear weights
        """
        
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)
        
        return x