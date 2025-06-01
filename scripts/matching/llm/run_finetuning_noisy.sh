#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

#seeds=(1924 1913 1941 1830 1204)
seeds=(1924)


#--model "llama3-70b-8192" \
#--model "qwen2.5:32b" \
#--endpoint "http://localhost:11435/v1"
#--endpoint "http://gaia-gpu-2.imsi.athenarc.gr:11434/v1"

<<xom
for dir in "${directories[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Processing directory: $dir with seed: $seed"
       

        python ../build_prompt.py \
        --dataset "$dir" \
        --out_file "../../../log/matching/finetuning/noisy_qwen_14/partial/${dir}_${seed}.json"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
        --seed $seed \
        --serialization "DITTO" \
        --task_description "EXPLAIN"
        
        python ../run_prompt.py \
            --dataset "$dir" \
            --model "qwen2.5:14b" \
            --in_file "../../../log/matching/finetuning/noisy_qwen_14/partial/${dir}_${seed}.json" \
            --out_file "../../../log/matching/finetuning/noisy_qwen_14/partial_responses/${dir}_${seed}_responses.json" \
            --endpoint "http://localhost:11434/v1"


        python ../embed_noisy.py \
        --prompts "../../../log/matching/finetuning/noisy_qwen_14/partial/${dir}_${seed}.json"  \
        --labels "../../../log/matching/finetuning/noisy_qwen_14/partial_responses/${dir}_${seed}_responses.json" \
        --out_file "../../../log/matching/finetuning/noisy_qwen_14/partial_noisy/${dir}_${seed}.json" 
    done
done

xom

input_files=(../../../log/matching/finetuning/noisy_qwen_14/partial_noisy/*.json)
python ../fine_tuning_data.py \
   --input_files "${input_files[@]}" \
   --out_file "../../../log/matching/finetuning/noisy_qwen_14/train.json" \
   --mode "train" \
   --field "noise_answer"

python ../fine_tuning_train.py \
   --input_file "../../../log/matching/finetuning/noisy_qwen_14/train.json" \
   --log_file "../../../log/matching/finetuning/noisy_qwen_14/train_log.json" \
   --out_dir "../../../log/matching/finetuning/noisy_qwen_14/llama31_gt" \
   --out_name "llama31_gt" \
   --model "llama3.1"


for dir in "${directories[@]}"; do
    echo "Processing directory: $dir with seed: $seed"
   
    python ../build_prompt.py \
    --dataset "$dir" \
    --out_file "../../../log/matching/finetuning/noisy_qwen_14/test/${dir}_1924.json"  \
    --in_dir "../../../data/ccer/cleaned/original/" \
    --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
    --seed 1924 \
    --serialization "DITTO" \
    --task_description "EXPLAIN"
    
    python ../fine_tuning_data.py \
       --input_files "../../../log/matching/finetuning/noisy_qwen_14/test/${dir}_1924.json" \
       --out_file "../../../log/matching/finetuning/noisy_qwen_14/test/${dir}_total.json" \
       --mode "test"
    
    python ../fine_tuning_test.py \
       --dataset "$dir" \
       --model_path "../../../log/matching/finetuning/noisy_qwen_14/llama31_gt" \
       --input_file "../../../log/matching/finetuning/noisy_qwen_14/test/${dir}_total.json" \
       --out_file "../../../log/matching/finetuning/noisy_qwen_14/test_responses/${dir}_responses.json"
done