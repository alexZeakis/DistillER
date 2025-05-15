#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

#seeds=(1924 1913 1941 1830 1204)
seeds=(1924)

<<xom
for dir in "${directories[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Processing directory: $dir with seed: $seed"
       
        python ../build_prompt.py \
        --dataset "$dir" \
        --out_file "../../../log/matching/finetuning/ground/partial/${dir}_${seed}.json"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
        --seed $seed \
        --serialization "DITTO" \
        --task_description "EXPLAIN"
    done
done

input_files=(../../../log/matching/finetuning/ground/partial/*.json)
python ../fine_tuning_data.py \
   --input_files "${input_files[@]}" \
   --out_file "../../../log/matching/finetuning/ground/train.json" \
   --mode "train"


python ../fine_tuning_train.py \
   --input_file "../../../log/matching/finetuning/ground/train.json" \
   --log_file "../../../log/matching/finetuning/ground/train_log.json" \
   --out_dir "../../../log/matching/finetuning/ground/llama31_gt" \
   --out_name "llama31_gt" \
   --model "llama3.1"
xom


for dir in "${directories[@]}"; do
    echo "Processing directory: $dir with seed: $seed"
   
    python ../build_prompt.py \
    --dataset "$dir" \
    --out_file "../../../log/matching/finetuning/ground/test/${dir}_1924.json"  \
    --in_dir "../../../data/ccer/cleaned/original/" \
    --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
    --seed 1924 \
    --serialization "DITTO" \
    --task_description "EXPLAIN"
    
    python ../fine_tuning_data.py \
       --input_files "../../../log/matching/finetuning/ground/test/${dir}_1924.json" \
       --out_file "../../../log/matching/finetuning/ground/test/${dir}_total.json" \
       --mode "test"
    
    python ../fine_tuning_test.py \
       --dataset "$dir" \
       --model_path "../../../log/matching/finetuning/ground/llama31_gt" \
       --input_file "../../../log/matching/finetuning/ground/test/${dir}_total.json" \
       --out_file "../../../log/matching/finetuning/ground/test_responses/${dir}_responses.json"
done