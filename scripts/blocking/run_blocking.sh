#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9" )

# Loop over each directory and run the Python script
for dir in "${directories[@]}"; do
    echo "Processing directory: $dir"
<<xom    
    python vectorization.py \
    --dataset "$dir" \
    --model "sgtrt5" \
    --logfile "../../log/blocking/vectorization.txt" \
    --in_dir "../../data/ccer/cleaned/original/" \
    --out_dir "../../log/blocking/embeddings/" \
    --device "cuda:1"

    python blocking.py \
    --dataset "$dir" \
    --logdir "../../log/blocking/" \
    --in_dir "../../data/ccer/cleaned/original/"  \
    --emb_dir "../../log/blocking/embeddings/" \
    --device "cuda:1"  \
    --k 10
xom

    python sample_datasets.py \
    --dataset "$dir" \
    --in_dir "../../log/blocking/" \
    --out_dir "../../data/ccer/cleaned/fine_tuning/train/" \
    --blocking_option "intersection"  \
    --pos_number 300 \
    --neg_number 100

    python sample_datasets.py \
    --dataset "$dir" \
    --in_dir "../../log/blocking/" \
    --out_dir "../../data/ccer/cleaned/fine_tuning/test/" \
    --existing_file "../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
    --blocking_option "intersection"  \
    --pos_number 300 \
    --neg_number 100
    
done
