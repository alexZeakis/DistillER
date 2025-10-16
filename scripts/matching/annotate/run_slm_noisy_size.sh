#!/bin/bash

# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Seeds
seeds=(1924)

noisy="qwen_32"

# SLM models and corresponding paths
slm_models=("sminilm" "roberta")
slm_paths=("sentence-transformers/all-MiniLM-L6-v2" "roberta-base")

sizes=(0.95 0.90 0.80)

for size in "${sizes[@]}"; do
    for i in "${!slm_models[@]}"; do
        slm_model="${slm_models[$i]}"
        slm_path="${slm_paths[$i]}"
    
        echo "Running with SLM model: $slm_model, path: $slm_path"
    
        # Step 1: Transform data for each directory and seed
        for dir in "${directories[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Transforming data for $dir with seed $seed"
    
                python ../slm/transform_data_plm.py \
                    --dataset "$dir" \
                    --out_dir "../../../log/matching/annotate/slm/size/$slm_model/${size}/data/$dir/"  \
                    --in_dir "../../../data/ccer/cleaned/original/" \
                    --sample_file "../../../data/ccer/cleaned/fine_tuning/blocking/train/$dir.csv" \
                    --seed $seed \
                    --serialization "DITTO" \
                    --percentage ${size} \
                    --mode "train" \
                    --response_file "../../../log/matching/annotate/blocking/${noisy}/partial_responses/${dir}_${seed}_responses.json"
            done
        done
    
        # Step 2: Merge transformed data
        python merge_data.py \
            --in_dir "../../../log/matching/annotate/slm/size/$slm_model/${size}/data" \
            --out_dir "../../../log/matching/annotate/slm/size/$slm_model/${size}/data/total/" \
            --percentage 0.8
    
        # Step 3: Fine-tune the SLM model
        python ../slm/supervised_main.py \
            --model_type "$slm_model" \
            --model_name_or_path "$slm_path" \
            --data_dir "../../../log/matching/annotate/slm/size/$slm_model/${size}/data/" \
            --log_dir "../../../log/matching/annotate/slm/size/$slm_model/${size}/log/" \
            --exp_dir "../../../log/matching/annotate/slm/size/$slm_model/${size}/exp/" \
            --data_name "total" \
            --train_batch_size 16 \
            --eval_batch_size 16 \
            --num_epochs 15.0 \
            --seed 1924 \
            --device "cuda:0"
    
        # Step 4: Build prompt and embed noisy data
        for dir in "${directories[@]}"; do
            echo "Final processing for $dir with seed 1924"
    
            python ../build_prompt.py \
                --dataset "$dir" \
                --out_file "../../../log/matching/annotate/slm/size/$slm_model/${size}/partial/${dir}_1924.json" \
                --in_dir "../../../data/ccer/cleaned/original/" \
                --sample_file "../../../data/ccer/cleaned/fine_tuning/blocking/train/$dir.csv" \
                --seed 1924 \
                --serialization "DITTO" \
                --task_description "EXPLAIN"
    
            python ../embed_noisy.py \
                --prompts "../../../log/matching/annotate/slm/size/$slm_model/${size}/partial/${dir}_1924.json" \
                --labels "../../../log/matching/annotate/slm/size/$slm_model/${size}/log/total_predictions.csv" \
                --out_file "../../../log/matching/annotate/slm/size/$slm_model/${size}/partial_noisy/${dir}_1924.json"
        done
    done
done