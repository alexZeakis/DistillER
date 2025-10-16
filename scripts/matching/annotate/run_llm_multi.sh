#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Seeds
seeds=(1924)

# Models and corresponding output directories
out_dirs=("llama_8" "llama_70" "qwen_14" "qwen_32")

mkdir -p "../../../log/matching/annotate/llm/multi_llm/partial"

# Loop over models and corresponding output directories
for dir in "${directories[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Processing: dir=$dir, seed=$seed"

        cp "../../../log/matching/annotate/llm/llama_8/partial/${dir}_${seed}.json"  "../../../log/matching/annotate/llm/multi_llm/partial/${dir}_${seed}.json" 

        in_files=$(for o in "${out_dirs[@]}"; do echo -n "../../../log/matching/annotate/llm/${o}/partial_responses/${dir}_${seed}_responses.json "; done)

        python merge_responses.py \
            --in_files $in_files \
            --out_file "../../../log/matching/annotate/llm/multi_llm/partial_responses/${dir}_${seed}_responses.json" \

        python ../embed_noisy.py \
            --prompts "../../../log/matching/annotate/llm/multi_llm/partial/${dir}_${seed}.json" \
            --labels "../../../log/matching/annotate/llm/multi_llm/partial_responses/${dir}_${seed}_responses.json" \
            --out_file "../../../log/matching/annotate/llm/multi_llm/partial_noisy/${dir}_${seed}.json"
    done
done
