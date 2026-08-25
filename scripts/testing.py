from time import sleep
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from docudino.model import dino_v1
from docudino.data import DocumentDataset, DocumentSampler, TrainingAugmentations

if __name__ == "__main__":
    # load training data
    transform = TrainingAugmentations(
        (0.40, 1.00),
        (0.10, 0.40),
        8,
    )
    dataset = DocumentDataset("datasets/historical_wi/train", 256, 256, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        sampler=DocumentSampler(dataset),
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    
    # load model
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = dino_v1.vit_small(d_head=8192, training=True).to(DEVICE)
    
    for images in tqdm(dataloader):
        images: torch.Tensor
        images = [img.to(DEVICE, non_blocking=True) for img in images]
        
        tokens: torch.Tensor = model(images)
        
        print(tokens, type(tokens))
    
    print("Done processing")