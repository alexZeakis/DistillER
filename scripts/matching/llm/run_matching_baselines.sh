#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9" )

for dir in "${directories[@]}"; do
    echo "Processing directory: $dir"

    python umc.py \
     --in_file "../../../data/ccer/fine_tuning/test/$dir.csv" \
     --out_file "../../../log/matching/baselines/umc/$dir.csv"
    
done
