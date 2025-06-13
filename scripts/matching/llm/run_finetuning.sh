#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

models=("ground" "llama_8" "llama_70" "qwen_14" "qwen_32"  \
         "sminilm/llama_70" "roberta/llama_70")

for noisy in "${models[@]}"; do
    
    input_files=(../../../log/matching/annotate/$noisy/partial_noisy/*.json)
    python fine_tuning_data.py \
       --input_files "${input_files[@]}" \
       --out_file "../../../log/matching/llm/$noisy/train.json" \
       --mode "train" \
       --field "noise_answer"
    
    python fine_tuning_train.py \
       --input_file "../../../log/matching/llm/$noisy/train.json" \
       --log_file "../../../log/matching/llm/$noisy/train_log.json" \
       --out_dir "../../../log/matching/llm/$noisy/llama31_gt" \
       --out_name "llama31_gt" \
       --model "llama3.1"
    
    
    for dir in "${directories[@]}"; do
        echo "Processing directory: $dir with seed: $seed"
       
        python ../build_prompt.py \
        --dataset "$dir" \
        --out_file "../../../log/matching/llm/$noisy/test/${dir}_1924.json"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
        --seed 1924 \
        --serialization "DITTO" \
        --task_description "EXPLAIN"
        
        python fine_tuning_data.py \
           --input_files "../../../log/matching/llm/$noisy/test/${dir}_1924.json" \
           --out_file "../../../log/matching/llm/$noisy/test/${dir}_total.json" \
           --mode "test"
        
        python fine_tuning_test.py \
           --dataset "$dir" \
           --model_path "../../../log/matching/llm/$noisy/llama31_gt" \
           --input_file "../../../log/matching/llm/$noisy/test/${dir}_total.json" \
           --out_file "../../../log/matching/llm/$noisy/test_responses/${dir}_responses.json"
    done
done    