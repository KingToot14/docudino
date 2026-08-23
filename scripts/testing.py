from time import sleep
from tqdm import tqdm

from torch.utils.data import DataLoader

from docudino.data import DocumentDataset, DocumentSampler

if __name__ == "__main__":
    dataset = DocumentDataset("datasets/historical_wi/train", 256, 256)
    
    dataloader = DataLoader(
        dataset,
        sampler=DocumentSampler(dataset)
    )
    
    for image in tqdm(dataloader):
        pass
    
    print("Done processing")