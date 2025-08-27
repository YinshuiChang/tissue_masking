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

def threshold_otsu(array: np.ndarray, nbins: int = 100) -> float:
    """
    Parameters
    ----------
    array
        Image to be binarized.
    nbins
        Number of bins to use in the binarization histogram.

    Returns
    -------
    Binarization threshold.

    Reference
    ---------
    Otsu, N., 1979. A threshold selection method from gray-level histograms.
    IEEE transactions on systems, man, and cybernetics, 9(1), pp.62-66.

    """
    array = array.ravel().flatten()
    hist, bin_edges = np.histogram(array, bins=nbins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist.astype(float)
    # Class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def binerized_image(array: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    """
    Parameters
    ----------
    array
        Image to be binarized.
    method
        Binarization method, either 'adaptive' or 'otsu'.

    Returns
    -------
    Binarized image.

    """
    if method == 'adaptive':
        array = array / array.max() *255
        array = cv2.adaptiveThreshold(
            array.astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    elif method == 'otsu':
        thresh = threshold_otsu(array)
        array = (array > thresh).astype(np.uint8)
    else:
        raise ValueError("Method must be either 'adaptive' or 'otsu'.")
    return array

def merge_HAE_channels(array: np.ndarray) ->  np.ndarray:
    """
    Parameters
    ----------
    array
        H&E Image with RGB(CHW).

    Returns
    -------
    artefact removed single channel H&E Image

    Reference
    ---------
    Schreiber, B.A., Denholm, J., Jaeckle, F. et al. 
    Rapid artefact removal and H&E-stained tissue segmentation. 
    Sci Rep 14, 309 (2024). https://doi.org/10.1038/s41598-023-50183-4
    """
    
    array = array / 255
    RG = np.maximum(array[0,:,:] - array[1,:,:], 0)
    BG = np.maximum(array[2,:,:] - array[1,:,:], 0)
    array = RG*BG
    return array

def noise_removal(barray: np.ndarray) ->  np.ndarray:
    """
    Parameters
    ----------
    array
        binerized Image(uint8).

    Returns
    -------
    noise removed binerized Image
    
    """
    
    kernel_round = np.ones((5, 5), np.uint8)
    kernel_round[0][0] = 0
    kernel_round[0][4] = 0
    kernel_round[4][0] = 0
    kernel_round[4][4] = 0
    
    # Morph close
    barray = cv2.dilate(barray, kernel_round, iterations=2)
    barray = cv2.erode(barray, kernel_round, iterations=2)
    # Morph open
    barray = cv2.erode(barray, kernel_round, iterations=1)
    barray = cv2.dilate(barray, kernel_round, iterations=1)
    return barray

def tissue_identification(barray: np.ndarray, min_size_tissue = 256*256/4, min_size_hole = 256*256/4) ->  np.ndarray:
    """
    Parameters
    ----------
    array
        binerized Image(uint8)
    min_size_tissue
        

    Returns
    -------
    Tissue selection (RGB)
        R: out of Tissue cells
        G: Tissue
        B: Holes
    
    """

    contours, hierarchy = cv2.findContours(
        barray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )
    tissue_idx = []
    hole_idx = []
    canvas = np.zeros((*barray.shape,3))
    for idx, (cnt, hier) in enumerate(zip(contours, hierarchy[0])):
        if hier[3] == -1 or hier[3] not in tissue_idx:
            if cv2.contourArea(cnt) > min_size_tissue:
                cv2.drawContours(canvas, [cnt], -1, (0, 1, 0), -1)
                tissue_idx.append(idx)
            else:
                if hier[3] in hole_idx and cv2.contourArea(cnt) > min_size_tissue/10:
                    cv2.drawContours(canvas, [cnt], -1, (0, 1, 0), -1)
                    tissue_idx.append(idx)
                else:
                    cv2.drawContours(canvas, [cnt], -1, (1, 0, 0), -1)
        else:
            if cv2.contourArea(cnt) > min_size_hole:
                cv2.drawContours(canvas, [cnt], -1, (0, 0, 1), -1)
                hole_idx.append(idx)
            else:
                cv2.drawContours(canvas, [cnt], -1, (0, 1, 0), -1)
                tissue_idx.append(idx)
    return canvas

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
        array = cv2.boxFilter(array[:,:],-1,ksize = (tile_size_px,tile_size_px),anchor = (0,0),normalize = True)
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
    
def write_metadata(hae_feature_absolute_path, idx_array_absolute_path, stride_size_px, tile_size_px, shape, dtype, filter_small = False, min_size_tissue = 256*256/4, min_size_hole = 256*256/16):
    meta = {
        'name': 'masked_index',
        'version': VERSION,
        'datas_source': str(hae_feature_absolute_path),
        'from': 'hematoxylin_eosin.zarr',
        'parameters': {
            'stride_size_px': stride_size_px,
            'tile_size_px': tile_size_px,
            'exclude_out_of_tissue': True,
            'shape': shape,
            'dtype': dtype,
            'chunks': (4096, 3),
            'shards': (4096*4096, 3),
            'filter_small_contour': filter_small,
            'min_size_tissue': min_size_tissue,
            'min_size_hole': min_size_hole,
        }
    }

    with open(f'{idx_array_absolute_path[:-5]}.json', 'w') as file:
        json.dump(meta, file)

def generate_idx_array_hae(hae_feature_absolute_path, idx_array_absolute_path, stride_size_px, tile_size_px = 256, filter_small = False, min_size_tissue = 256*256/4, min_size_hole = 256*256/16):
    hae_feature_absolute_path = Path(hae_feature_absolute_path)
    z = zarr.open(hae_feature_absolute_path / 'hematoxylin_eosin.zarr', mode='r')
    array = z[:,:,:]
    array = merge_HAE_channels(array)
    barray = binerized_image(array)
    bmask = noise_removal(barray)
    if filter_small:
        canvas = tissue_identification(bmask, min_size_tissue, min_size_hole)
        bmask = canvas[:,:,1]
    bmask = bmask.astype(np.float32)
    out = to_index_array(bmask, tile_size_px)
    write_zarr(out, idx_array_absolute_path)
    write_metadata(hae_feature_absolute_path, idx_array_absolute_path, stride_size_px, tile_size_px, out.shape, str(out.dtype), filter_small, min_size_tissue)

def generate_data_array_hae(hae_feature_absolute_path, out_array_absolute_path, stride_size_px, tile_size_px = 256, filter_small = False, min_size_tissue = 256*256/4, min_size_hole = 256*256/16):
    hae_feature_absolute_path = Path(hae_feature_absolute_path)
    out_array_absolute_path = Path(out_array_absolute_path)
    z = zarr.open(hae_feature_absolute_path / 'hematoxylin_eosin.zarr', mode='r')
    array = z[:,:,:]
    array = merge_HAE_channels(array)
    barray = binerized_image(array)
    bmask = noise_removal(barray)
    if filter_small:
        canvas = tissue_identification(bmask, min_size_tissue, min_size_hole)
        canvas.astype(np.float32)
        write_zarr(canvas, out_array_absolute_path / 'v2_tissue_mask_HAE_stride_256_tile_256.zarr', chunks = (256, 256, 3), shards = (256*4, 256*4, 3))
        bmask = canvas[:,:,1]
    # out = to_index_array(bmask, tile_size_px)
    # out = cv2.boxFilter(bmask[:,:],-1,ksize = (tile_size_px,tile_size_px),anchor = (0,0),normalize = False)
    # write_zarr(out, out_array_absolute_path / 'v2_data_array_HAE_mask_stride_256_tile_256.zarr', chunks = (256, 256), shards = (256*4, 256*4))
    # write_metadata(hae_feature_absolute_path, str(out_array_absolute_path / 'v2_data_array_HAE_mask_stride_256_tile_256.zarr'), stride_size_px, tile_size_px, out.shape, str(out.dtype), filter_small, min_size_tissue)



if __name__ == '__main__':
    fire.Fire()