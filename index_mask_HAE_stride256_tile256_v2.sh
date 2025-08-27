#!/bin/bash 

DIR="/lustre/scratch127/cellgen/cellgeni/tickets/tic-4033/masked_index_scripts"
SOURCE="/lustre/scratch126/cellgen/lotfollahi/projects/histology_to_gene_expression/data_release_v2.0/feature"
OUT="/lustre/scratch126/cellgen/lotfollahi/projects/histology_to_gene_expression/masked_index_test_v2"
SL=$1

mkdir -p $OUT/$SL
python $DIR/index_masking_HAE.py generate_idx_array_hae \
    $SOURCE/$SL \
    $OUT/$SL/index_mask_HAE_stride_256_tile_256_v2.zarr 256 256 True