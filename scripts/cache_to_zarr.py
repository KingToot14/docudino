from argparse import ArgumentParser

from docudino.evaluation.config import load_evaluation_config
from docudino.data.dataset import create_evaluation_dataloader

if __name__ == "__main__":
    # parse config file location
        parser = ArgumentParser(
            "Cache Dataset to zarr",
            description="Caches an on-demand dataset into a pre-computed zarr dataset",
        )
        
        parser.add_argument(
            "config", help="The location of the config file to load for fetching data"
        )
        parser.add_argument(
            "output", help="The location to save the zarr dataset to"
        )
        
        args, overrides = parser.parse_known_args()
        
        # load config file
        cfg = load_evaluation_config(args.config, overrides)
        
        # load dataset
        dataloader = create_evaluation_dataloader(cfg, True, 0, 1)
        
        for batch in dataloader:
            windows, writers, documents = batch
            
            print(windows.shape)
            print(writers)
            print(documents)