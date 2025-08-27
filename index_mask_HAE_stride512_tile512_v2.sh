#!/bin/bash 

DIR="/lustre/scratch127/cellgen/cellgeni/tickets/tic-4033/masked_index_scripts"
SOURCE="/nfs/team361/data-perprocessing/data_release_v2.0/feature"
OUT="/lustre/scratch126/cellgen/lotfollahi/projects/histology_to_gene_expression/masked_index_test_v2"
SL=$1

mkdir -p $OUT/$SL
python $DIR/index_masking_HAE.py generate_idx_array_hae \
    $SOURCE/$SL \
    $OUT/$SL/index_mask_HAE_stride_512_tile_512_v2.zarr 512 512 True