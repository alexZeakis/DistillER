#!/bin/bash

# List of directories
#directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9" )
#directories=("D5")
directories=("D3" "D4" "D6" "D7" "D8" "D9" )

# Loop over each directory and run the Python script
for dir in "${directories[@]}"; do
    echo "Processing directory: $dir"
    
    python ../build_prompt.py \
    --dataset "$dir" \
    --out_file "../../../log/matching/baselines/pretrained/$dir.json"  \
    --in_dir "../../../data/ccer/cleaned/original/" \
    --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
    --seed 1924 \
    --serialization "DITTO" \
    --task_description "EXPLAIN"

    python ../run_prompt.py \
        --dataset "$dir" \
        --model "llama3.1:latest" \
        --in_file "../../../log/matching/baselines/pretrained/$dir.json" \
        --out_file "../../../log/matching/baselines/pretrained/${dir}_responses.json" \
        --endpoint "http://localhost:11435/v1"
        #--token "ollama"
        #--model "qwen3:8b" \
done