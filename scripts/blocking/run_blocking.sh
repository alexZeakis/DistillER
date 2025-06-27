#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9" )

methods=("random" "blocking" "sampled")

# Loop over each directory and run the Python script
for dir in "${directories[@]}"; do
    echo "Processing directory: $dir"

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

    for method in "${methods[@]}"; do
        python split_dataset.py \
        --dataset "$dir" \
        --in_dir "../../log/blocking/" \
        --out_dir "../../data/ccer/cleaned/fine_tuning/$method/" \
        --blocking_option "intersection"  \
        --split_percent 0.1 \
        --method "$method" \
        --positive_ratio 0.75

    done
done
