from time import sleep
from tqdm import tqdm

from docudino.data import DocumentDataset

if __name__ == "__main__":
    dataset = DocumentDataset("datasets/historical_wi/train", 256, 256)
    
    for image in tqdm(dataset):
        pass
    
    print("Done processing")