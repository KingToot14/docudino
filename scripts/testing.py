from time import sleep
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from docudino.transformer import dino_v1
from docudino.data import DocumentDataset, DocumentSampler

if __name__ == "__main__":
    # load training data
    dataset = DocumentDataset("datasets/historical_wi/train", 256, 256)
    
    dataloader = DataLoader(
        dataset,
        sampler=DocumentSampler(dataset),
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    
    # load model
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = dino_v1.vit_small().to(DEVICE)
    
    for image in tqdm(dataloader):
        image: torch.Tensor
        image = image.to(DEVICE, non_blocking=True)
        
        tokens: torch.Tensor = model(image)
    
    print("Done processing")