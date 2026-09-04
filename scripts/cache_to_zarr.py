from argparse import ArgumentParser

import zarr
from zarr.codecs import BloscCodec

import numpy as np
import torchvision.transforms.v2 as T

from tqdm import tqdm

from docudino.evaluation.config import load_evaluation_config
from docudino.data.dataset import create_evaluation_dataloader


def cache_dataset(config_file: str, overrides: list[str], output: str, is_train: bool):
    # load config file
    cfg = load_evaluation_config(config_file, overrides)
    
    # load dataset
    dataloader = create_evaluation_dataloader(cfg, is_train, 0, 1)
    dataloader.dataset.transform = T.ToImage()
    total_windows = len(dataloader.dataset)
    
    compressors = BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')
    
    # create zarr dataset
    root = zarr.create_group(
        f"{output}/{"train" if is_train else "test"}",
        attributes={
            "window_size": cfg.extract.window_size,
            "is_training": is_train,
        },
        overwrite=True,
    )
    
    images = root.create_array(
        name="images",
        shape=(total_windows, 3, cfg.extract.window_size, cfg.extract.window_size),
        chunks=(256, 3, cfg.extract.window_size, cfg.extract.window_size),
        dtype=np.int8,
        overwrite=True,
        compressors=compressors,
    )
    doc_ids = root.create_array(
        name="doc_ids",
        shape=(total_windows,),
        chunks=(4096,),
        dtype=np.int32,
        overwrite=True,
        compressors=compressors,
    )
    doc_map = root.create_array(
        name="doc_map",
        shape=(dataloader.dataset.image_count,),
        dtype=np.int32,
        overwrite=True,
        compressors=compressors,
    )
    
    batch_size = cfg.dataset.batch_size
    
    # fill dataset
    for i, batch in enumerate(tqdm(dataloader)):
        windows, _, documents = batch
        
        # write to dataset
        start_idx = i * batch_size
        
        images[start_idx : start_idx + batch.shape[0]] = windows.numpy()
        doc_ids[start_idx : start_idx + batch.shape[0]] = documents.numpy()
    
    # fill writer -> document map
    for writer, document in dataloader.dataset.image_info:
        doc_map[writer] = document

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
        
        cache_dataset(args.config, overrides, args.output, True)
        cache_dataset(args.config, overrides, args.output, False)