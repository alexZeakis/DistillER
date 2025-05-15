#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

#seeds=(1924 1913 1941 1830 1204)
seeds=(1924)


#--model "llama3.1:70b" \
#--model "llama3-70b-8192" \
#--model "qwen2.5:32b" \
#--endpoint "http://gaia-gpu-2.imsi.athenarc.gr:11434/v1"

<<xom
for dir in "${directories[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Processing directory: $dir with seed: $seed"
       
        python ../build_prompt.py \
        --dataset "$dir" \
        --out_file "../../../log/matching/confidence/noisy/partial/${dir}_${seed}.json"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
        --seed $seed \
        --serialization "DITTO" \
        --task_description "CONFIDENCE"
        
        python ../run_prompt.py \
            --dataset "$dir" \
            --model "llama3.1:latest" \
            --in_file "../../../log/matching/confidence/noisy/partial/${dir}_${seed}.json" \
            --out_file "../../../log/matching/confidence/noisy/partial_responses/${dir}_${seed}_responses.json" \
            --endpoint "http://localhost:11434/v1"

        python ../embed_noisy.py \
        --prompts "../../../log/matching/confidence/noisy/partial/${dir}_${seed}.json"  \
        --labels "../../../log/matching/confidence/noisy/partial_responses/${dir}_${seed}_responses.json" \
        --out_file "../../../log/matching/confidence/noisy/partial_noisy_clean/${dir}_${seed}.json" \
        --confidence True \
        --confidence_threshold 0.9

    done
done



input_files=(../../../log/matching/confidence/noisy/partial_noisy_clean/*.json)
python ../fine_tuning_data.py \
   --input_files "${input_files[@]}" \
   --out_file "../../../log/matching/confidence/noisy/train.json" \
   --mode "train" \
   --field "noise_answer"



python ../fine_tuning_train.py \
   --input_file "../../../log/matching/confidence/noisy/train.json" \
   --log_file "../../../log/matching/confidence/noisy/train_log.json" \
   --out_dir "../../../log/matching/confidence/noisy/llama31_gt" \
   --out_name "llama31_gt" \
   --model "llama3.1"
xom

for dir in "${directories[@]}"; do
    echo "Processing directory: $dir with seed: $seed"
   
    python ../build_prompt.py \
    --dataset "$dir" \
    --out_file "../../../log/matching/confidence/noisy/test/${dir}_1924.json"  \
    --in_dir "../../../data/ccer/cleaned/original/" \
    --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
    --seed 1924 \
    --serialization "DITTO" \
    --task_description "EXPLAIN"
    
    python ../fine_tuning_data.py \
       --input_files "../../../log/matching/confidence/noisy/test/${dir}_1924.json" \
       --out_file "../../../log/matching/confidence/noisy/test/${dir}_total.json" \
       --mode "test"
    
    python ../fine_tuning_test.py \
       --dataset "$dir" \
       --model_path "../../../log/matching/confidence/noisy/llama31_gt" \
       --input_file "../../../log/matching/confidence/noisy/test/${dir}_total.json" \
       --out_file "../../../log/matching/confidence/noisy/test_responses/${dir}_responses.json"
done
