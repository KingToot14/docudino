import csv
from pathlib import Path

import torch
from torch import Tensor

from .index import WriterIndex

class Metrics:
    def __init__(self, descriptors: Tensor, writers: Tensor, index: WriterIndex):
        self.descriptors = descriptors
        self.writers = writers
        
        # create writer mapping
        self.writer_map: dict[int, Tensor] = {
            writer.item(): torch.where(writers == writer)[0]
            for writer in writers.unique()
        }
        
        self.index = index
    
    def top_k_accuracy(self, k: int) -> float:
        """
        Calculates and returns the top-k accuracy over all descriptors
        """
        
        _, indices = self.index.get_top_k(self.descriptors, k)
        
        retrieved = self.writers[indices]
        queried = self.writers.unsqueeze(1)
        
        correct = (retrieved == queried).any(dim=1)
        
        return correct.float().mean().item()
    
    def mean_average_precision(self, k: int = -1) -> float:
        """
        Calculates and returns the mean average precision. `k` is optional and specifies the maximum
        number of documents to consider "hits"
        """
        
        # get up to k matches (or all if k == -1)
        _, indices = self.index.get_top_k(self.descriptors, k)
        
        # get labels
        retrieved = self.writers[indices]
        queried = self.writers[:, None]
        
        # get hits over time
        relevant = retrieved == queried
        cumulative_hits = relevant.cumsum(dim=1)
        
        ranks = torch.arange(
            1, indices.shape[1] + 1,
            device=indices.device
        )
        
        # calculate average precision only at hits (where relevant is True)
        precision_at_rank = cumulative_hits / ranks
        ap = (precision_at_rank * relevant).sum(dim=1)
        
        # calculate the number of relevant documents per-query (minus self query)
        num_relevant = torch.tensor(
            [
                len(self.writer_map[self.writers[i].item()]) - 1
                for i in range(len(self.writers))
            ]
        )
        
        # calculate and return mAP
        ap /= num_relevant
        return ap.mean().item()
    
    def run_metrics(self, out_path: str, name: str, metrics: list[str]) -> None:
        """
        Calculates a set of `metrics` with a few configurations in the form: <metric>:<k>. These
        results are output to a file in `out_path` in a csv format
        
        `metrics` can be among: [topk:k, mAP:k]. Note, if k == -1, k will be omitted when saving
        """
        
        labels: list[str] = ["name"]
        outputs: dict[str, float] = {}
        
        # add name
        outputs["name"] = name
        
        # calculate metrics
        for metric in metrics:
            method: str
            k: int
            
            if ':' in metric:
                method, k = metric.split(':')
                k = int(k)
            else:
                method = metric
                k = -1
            
            if k == -1:
                labels.append(method)
            else:
                labels.append(f"{method}@{k}")
            
            match method:
                case 'topk':
                    outputs[labels[-1]] = self.top_k_accuracy(k)
                case 'mAP':
                    outputs[labels[-1]] = self.mean_average_precision(k)
            
            print(labels[-1], outputs[labels[-1]])
        
        # make sure path exists
        Path("/".join(out_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        
        # append resutls
        with open(out_path, 'a') as fs:
            writer = csv.DictWriter(fs, fieldnames=labels)
            writer.writerow(outputs)