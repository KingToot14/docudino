import math

import torch
import torch.nn as nn

import torchvision.transforms.v2 as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST

from torch.utils.data import DataLoader

from tqdm import tqdm

class PatchEmbeddings(nn.Module):
    def __init__(self, d_model: int, patch_size: int, n_channels: int):
        """
        Creates a new `nn.Module` that embeds patches from an image
        
        Args:
            d_model (int): The dimensionality of the model
            patch_size (int): The size of the patches (square)
            n_channels (int): The number of channels passed into the transformer
        """
        super().__init__()
        
        # store parameters
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        # create a blank classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # create the patch splitter
        self.linear_proj = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
        
        # initialize parameters
        nn.init.trunc_normal_(self.cls_token, std=.02)
    
    def forward(self, x: torch.Tensor):
        """
        Runs a forward pass of the patch embedder module. A few important values:
          - B: Batch Size
          - C: Color Channels
          - H: Image Height
          - W: Image Width
          - P_col: Patch columns
          - P_row: Patch rows
        """
        
        # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = self.linear_proj(x)
        
        # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.flatten(2)
        
        # (B, d_model, P) -> (B, P, d_model)
        x = x.transpose(1, 2)
        
        # add class token
        tokens_batch = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((tokens_batch, x), dim=1)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, img_size: int, patch_size: int):
        """
        Creates a new `nn.Module` that appends a class token and positional encoding to a
        sequence of patches.
        
        Args:
            d_model (int): The dimensionality of the model
            img_size (int): The size of the image to be processed
            patch_size (int): The size of the patches taken from the image
        """
        
        super().__init__()
        
        # store parameters
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        
        # create a blank positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, d_model))
        
        # initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x: torch.Tensor, w: int, h: int):
        n_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if n_patches == N and w == h:
            return x + self.pos_embed
        
        # interpolate position embeds | x: (batches, patches, dimensions)
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        
        # apparently this fixes something with floating point precision
        w0 = w // self.patch_size + 0.1
        h0 = h // self.patch_size + 0.1
        
        patch_pos_embed = nn.functional.interpolate(
            # (batch, patch, dim) -> (batch, w, h, dim) -> (batch, dim, w, h)
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic'
        )
        
        # (batch, dim, w, h) -> (batch, w, h, dim) -> (batch, patch, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, head_size: int):
        """
        Creates a new `nn.Module` that implements scaled dot-product attention
        
        Args:
            d_model (int): The dimensionality of the model
            head_size (int): the size of each head
        """
        
        super().__init__()
        
        # store parameters
        self.head_size = head_size
        
        # create projections
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
    
    def forward(self, x: torch.Tensor):
        # collect query, key, and value
        Q: torch.Tensor = self.query(x)
        K: torch.Tensor = self.key(x)
        V: torch.Tensor = self.value(x)
        
        # calculate attention score
        score = Q @ K.transpose(-2, -1)
        
        # scaling
        attn = score / (self.head_size ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        
        # return the weighted sum of values
        return attn @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        Creates a new `nn.Module` that implements multi-head attention by running multiple
        instances of `AttentionHead` in sequence.
        
        Args:
            d_model (int): The dimensionality of the model
            n_heads (int): The number of heads to create
        """
        
        super().__init__()
        
        # store parameters
        assert d_model % n_heads == 0, "Model dimensionality must be divisible by the number of heads"
        self.head_size = d_model // n_heads
        
        # concatenation projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # create list of heads
        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])
    
    def forward(self, x: torch.Tensor):
        # combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        
        # learnable projection
        out = self.W_o(out)
        
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, r_mlp: int = 4):
        """
        Creates a new `nn.Module` that performs Transformer encoding with a pre-LN setup and
        `MultiHeadAttention` and `MLP` sublayers
        
        Args:
            d_model (int): The dimensionality of the model
            n_heads (int): The number of heads to create in the `MultiHeadAttention` block
            r_mlp (int): How many dimensions the `MLP` should multiply by. Default = 4
        """
        
        super().__init__()
        
        # store parameters
        self.d_model = d_model
        self.n_heads = n_heads
        
        # sublayer 1: Normalization
        self.ln1 = nn.LayerNorm(d_model)
        
        # sublayer 1: Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads)
        
        # sublayer 2: Normalization
        self.ln2 = nn.LayerNorm(d_model)
        
        # sublaye 3: Multi-layer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model),
        )
    
    def forward(self, x: torch.Tensor):
        # residual connection after sublayer 1
        x = x + self.mha(self.ln1(x))
        
        # residual connection after sublayer 2
        x = x + self.mlp(self.ln2(x))
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
            d_model: int,
            n_classes: int,
            img_size: int,
            patch_size: int,
            n_channels: int,
            n_heads: int,
            n_layers: int,
        ):
        
        super().__init__()
        
        # safety checks
        assert img_size % patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # store parameters
        self.d_model = d_model
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads
        
        # create encoders
        self.patch_embedding = PatchEmbeddings(d_model, patch_size, n_channels)
        self.positional_encoding = PositionalEncoding(d_model, img_size, patch_size)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
        
        # classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, images: torch.Tensor):
        B, C, W, H = images.shape
        
        # embded each patch
        x = self.patch_embedding(images)
        
        # add positional encoding
        x = self.positional_encoding(x, W, H)
        
        # run through transformer
        x = self.transformer_encoder(x)
        
        # classify image (using just the cls token)
        x = self.classifier(x[:,0])
        
        return x

if __name__ == "__main__":
    # training parameters
    d_model = 9
    n_classes = 10
    img_size = 32
    patch_size = 16
    n_channels = 1
    n_heads = 3
    n_layers = 3
    batch_size = 128
    epochs = 5
    alpha = 0.005
    
    # create image transform
    transform = T.Compose([
        T.Resize(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])
    
    # load MNIST
    train_set = MNIST(
        root="./datasets", train=True, download=True, transform=transform
    )
    test_set = MNIST(
        root="./datasets", train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_set, shuffle=True,  batch_size=batch_size)
    test_loader = DataLoader(test_set,   shuffle=False, batch_size=batch_size)
    
    # check training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    # create transformer
    transformer = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

    # create optimizer and loss
    optimizer = Adam(transformer.parameters(), lr=alpha)
    criterion = nn.CrossEntropyLoss()

    # start training
    for epoch in range(epochs):
        training_loss = 0.0
        
        # iterate through each batch
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # clear gradients
            optimizer.zero_grad()

            # run model, calculate loss
            outputs = transformer(inputs)
            loss = criterion(outputs, labels)
            
            # optimize model
            loss.backward()
            optimizer.step()

            # log time
            training_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_loader) :.2f}\n')
    
        # validation accuracy
        correct = 0
        total = 0
        
        # don't adjust gradients
        with torch.no_grad():
            for data in tqdm(test_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                # run transformer
                outputs = transformer(images)

                # check predictions
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Model Accuracy: {100.0 * correct / total:.2f}%\n')