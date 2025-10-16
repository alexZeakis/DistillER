#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Seeds
seeds=(1924)

# Models and corresponding output directories
models=("llama3.1:latest" "llama3.1:70b" "qwen2.5:14b" "qwen2.5:32b")
out_dirs=("llama_8" "llama_70" "qwen_14" "qwen_32")

# Loop over models and corresponding output directories
for i in "${!models[@]}"; do
    model="${models[$i]}"
    out_dir="${out_dirs[$i]}"

    # Set endpoint based on model
    if [[ "$model" == *"70b"* ]]; then
        endpoint="http://gaia-gpu-2.imsi.athenarc.gr:11434/v1"
    else
        endpoint="http://localhost:11434/v1"
    fi

    for dir in "${directories[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Processing: model=$model, dir=$dir, seed=$seed, endpoint=$endpoint"

            python ../build_prompt.py \
                --dataset "$dir" \
                --out_file "../../../log/matching/annotate/llm/$out_dir/partial/${dir}_${seed}.json" \
                --in_dir "../../../data/ccer/cleaned/original/" \
                --sample_file "../../../data/ccer/cleaned/fine_tuning/blocking/train/$dir.csv" \
                --seed $seed \
                --serialization "DITTO" \
                --task_description "EXPLAIN"

            python ../run_prompt.py \
                --dataset "$dir" \
                --model "$model" \
                --in_file "../../../log/matching/annotate/llm/$out_dir/partial/${dir}_${seed}.json" \
                --out_file "../../../log/matching/annotate/llm/$out_dir/partial_responses/${dir}_${seed}_responses.json" \
                --endpoint "$endpoint"

            python ../embed_noisy.py \
                --prompts "../../../log/matching/annotate/llm/$out_dir/partial/${dir}_${seed}.json" \
                --labels "../../../log/matching/annotate/llm/$out_dir/partial_responses/${dir}_${seed}_responses.json" \
                --out_file "../../../log/matching/annotate/llm/$out_dir/partial_noisy/${dir}_${seed}.json"
        done
    done
done
