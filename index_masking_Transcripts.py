import zarr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import dask.array as da
import json
import fire

VERSION = '0.0.1'

def to_index_array(array: np.ndarray, tile_size_px = 256, stride_size_px = 256) ->  np.ndarray:
    """
    Parameters
    ----------
    array
        tissue mask
    tile_size_px
        tile size in pixel

    Returns
    -------
    array:
        all FOVs with tissue mask
        0: x position
        1: y position
        2: area coverage of tissue mask 
    
    """
    def compute_block_sum(block):
        return np.array([np.sum(block)])[:, None]
    
    if tile_size_px == stride_size_px:
        h, w = array.shape
        array = array[:(h // tile_size_px)*tile_size_px, : (w // tile_size_px) * tile_size_px]
        array = da.from_array(array, chunks=(tile_size_px,tile_size_px))
        array = array.map_blocks(compute_block_sum, chunks=(1, 1)).compute()
        x, y = np.where(array > 0)
        x = x*tile_size_px
        y = y*tile_size_px
        values = array[array > 0]
        result = np.stack((x, y, values), axis=-1)
    else:
        array = cv2.boxFilter(array[:,:],-1,ksize = (tile_size_px,tile_size_px),anchor = (0,0),normalize = False)
        array = array[:(array.shape[0] + 1 - tile_size_px), :(array.shape[1] + 1 - tile_size_px)]
        x, y = np.where(array > 0)
        values = array[array > 0]
        result = np.stack((x, y, values), axis=-1)
        result = [i for i in result if i[0]%stride_size_px == 0 and i[1]%stride_size_px == 0] ## TODO: update striding logic!!! (use torch or keras)
        result = np.array(result)
    return result

def write_zarr(array: np.ndarray, idx_array_absolute_path, chunks = (4096, 3), shards = (4096*4096, 3)):
    """
    Parameters
    ----------
    array
        mask index array
    idx_array_absolute_path
        storage path
    chunks
        chunk size
    shards
        shard size (must be a multiple of chunk size)

    Returns
    -------
    None
    
    """
    
    shape = array.shape
    dtype = array.dtype

    slices_array = da.from_array(array, chunks=shards)
    
    zarr_array = zarr.create_array(
        store=idx_array_absolute_path,
        # mode="w",
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype
    )
    
    def write_block(block, zarr_array, block_info=None):
        if block_info is None:
            return da.array([[[0]]])
        slices = tuple(slice(start, stop) for start, stop in block_info[None]['array-location'])
        zarr_array[slices] = block
        return da.array([[[0]]])

    result = slices_array.map_blocks(write_block, zarr_array)
    result.compute()
    
def write_metadata(feature_absolute_path, idx_array_absolute_path, stride_size_px, tile_size_px, shape, dtype):
    meta = {
        'name': 'masked_index',
        'version': VERSION,
        'datas_source': str(feature_absolute_path),
        'from': 'hematoxylin_eosin.zarr',
        'parameters': {
            'stride_size_px': stride_size_px,
            'tile_size_px': tile_size_px,
            'exclude_out_of_tissue': True,
            'shape': shape,
            'dtype': dtype,
            'chunks': (4096, 3),
            'shards': (4096*4096, 3),
        }
    }

    with open(f'{idx_array_absolute_path[:-5]}.json', 'w') as file:
        json.dump(meta, file)

def generate_idx_array_transcripts(feature_absolute_path, idx_array_absolute_path, stride_size_px, tile_size_px = 256):
    feature_absolute_path = Path(feature_absolute_path)
    image_array = da.from_zarr(feature_absolute_path / 'spatially_binned_expression.zarr', chunks=(5001, 256, 256))
    results = image_array.sum(axis=0).compute()
    results = results.astype(np.int32)
    out = to_index_array(results, tile_size_px)
    write_zarr(out, idx_array_absolute_path)
    write_metadata(feature_absolute_path, idx_array_absolute_path, stride_size_px, tile_size_px, out.shape, str(out.dtype))

def generate_data_array_transcripts(feature_absolute_path, idx_array_absolute_path, stride_size_px, tile_size_px = 256):
    feature_absolute_path = Path(feature_absolute_path)
    image_array = da.from_zarr(feature_absolute_path / 'spatially_binned_expression.zarr', chunks=(5001, 256, 256))
    results = image_array.sum(axis=0).compute()
    results = results.astype(np.int32)
    out = cv2.boxFilter(results[:,:],-1,ksize = (tile_size_px,tile_size_px),anchor = (0,0),normalize = False)
    # out = to_index_array(results, tile_size_px)
    write_zarr(out, idx_array_absolute_path, chunks=(256, 256), shards=(256*4, 256*4))
    write_metadata(feature_absolute_path, idx_array_absolute_path, stride_size_px, tile_size_px, out.shape, str(out.dtype))



if __name__ == '__main__':
    fire.Fire()