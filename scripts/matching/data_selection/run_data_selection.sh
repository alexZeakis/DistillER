#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9" )

methods=("random" "sampled" "blocking_max" "blocking_top2" "clustering_kmeans" "clustering_hierarchical")

# Split Datasets
for dir in "${directories[@]}"; do
    echo "Processing directory: $dir"

    for method in "${methods[@]}"; do
        python split_dataset.py \
        --dataset "$dir" \
        --in_dir "../../../log/blocking/" \
        --out_dir "../../../data/ccer/cleaned/fine_tuning/$method/" \
        --blocking_option "intersection"  \
        --split_percent 0.1 \
        --method "$method" \
        --positive_ratio 0.75

    done
done


# Models and corresponding output directories
models=("llama3.1:latest" "qwen2.5:14b")
out_dirs=("llama_8" "qwen_14")

for strategy in "${methods[@]}"; do
    for i in "${!models[@]}"; do
        model="${models[$i]}"
        out_dir="${out_dirs[$i]}"
    
        for dir in "${directories[@]}"; do
            echo "Processing: model=$model, dir=$dir"

            python ../build_prompt.py \
                --dataset "$dir" \
                --out_file "../../../log/matching/data_selection/${strategy}/$out_dir/partial/${dir}_${seed}.json" \
                --in_dir "../../../data/ccer/cleaned/original/" \
                --sample_file "../../../data/ccer/cleaned/fine_tuning/${strategy}/train/$dir.csv" \
                --seed 1924 \
                --serialization "DITTO" \
                --task_description "EXPLAIN"

            python ../run_prompt.py \
                --dataset "$dir" \
                --model "$model" \
                --in_file "../../../log/matching/data_selection/${strategy}/$out_dir/partial/${dir}_${seed}.json" \
                --out_file "../../../log/matching/data_selection/${strategy}/$out_dir/partial_responses/${dir}_${seed}_responses.json" \
                --endpoint "http://localhost:11434/v1"

            python ../embed_noisy.py \
                --prompts "../../../log/matching/data_selection/${strategy}/$out_dir/partial/${dir}_${seed}.json" \
                --labels "../../../log/matching/data_selection/${strategy}/$out_dir/partial_responses/${dir}_${seed}_responses.json" \
                --out_file "../../../log/matching/data_selection/${strategy}/$out_dir/partial_noisy/${dir}_${seed}.json"
        done
    done
done  