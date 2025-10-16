#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

models=("qwen_32")

for noisy in "${models[@]}"; do

    if [ "$noisy" = "ground" ]; then
        input_files=(../../../log/matching/explanations/llm/$noisy/partial/*.json)
        field="ground_answer"
    elif [[ "$noisy" == llama* || "$noisy" == qwen* ]]; then
        explanation_files=(../../../log/matching/explanations/llm/$noisy/partial_responses/*.json)
        input_files=(../../../log/matching/explanations/llm/$noisy/partial_noisy/*.json)
        field="noise_answer"
    elif [[ "$noisy" == sminilm* || "$noisy" == roberta* ]]; then
        input_files=(../../../log/matching/explanations/llm/$noisy/partial_noisy/*.json)
        field="noise_answer"
    else
        echo "Unknown noisy value: $noisy"
        exit 1
    fi
 
    python ../rl/dpo_data.py \
       --input_files "${input_files[@]}" \
       --out_file "../../../log/matching/explanations/dpo/FT/$noisy/train.json" \
       --mode "train" \
       --field "${field}" \
       --explanation_files "${explanation_files[@]}" 

    
    python ../rl/train_dpo.py \
       --input_file "../../../log/matching/explanations/dpo/FT/$noisy/train.json" \
       --log_file "../../../log/matching/explanations/dpo/FT/$noisy/train_log.json" \
       --out_dir "../../../log/matching/explanations/dpo/FT/$noisy/llama31_gt" \
       --model "../../../log/matching/explanations/sft/qwen_32/llama31_gt"

    for dir in "${directories[@]}"; do
        echo "Processing directory: $dir with seed: $seed"

        python ../build_prompt.py \
        --dataset "$dir" \
        --out_file "../../../log/matching/explanations/dpo/FT/$noisy/test/${dir}_1924.json"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/blocking_max/test/$dir.csv" \
        --seed 1924 \
        --serialization "DITTO" \
        --task_description "EXPLAIN"
        
        python ../rl/dpo_data.py \
           --input_files "../../../log/matching/explanations/dpo/FT/$noisy/test/${dir}_1924.json" \
           --out_file "../../../log/matching/explanations/dpo/FT/$noisy/test/${dir}_total.json" \
           --mode "test"

        python ../rl/test_rl.py \
           --dataset "$dir" \
           --model_path "../../../log/matching/explanations/dpo/FT/$noisy/llama31_gt" \
           --input_file "../../../log/matching/explanations/dpo/FT/$noisy/test/${dir}_total.json" \
           --out_file "../../../log/matching/explanations/dpo/FT/$noisy/test_responses/${dir}_responses.json" 
           
    done
done