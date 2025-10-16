# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Noisy levels
noisy_levels=("ground" "qwen_32" "roberta/qwen_32")

slm_models=("sminilm" "roberta")
slm_paths=("sentence-transformers/all-MiniLM-L6-v2" "roberta-base")

for i in "${!slm_models[@]}"; do
    slm_model="${slm_models[$i]}"
    slm_path="${slm_paths[$i]}"
    
    # Loop over each noisy level
    for noisy in "${noisy_levels[@]}"; do
        echo "=== Processing: $noisy ==="
        
        python supervised_train.py \
            --model_type "$slm_model" \
            --model_name_or_path "$slm_path" \
            --data_dir "../../../../log/matching/sft/slm/data/${noisy}/data/"   \
            --log_dir "../../../../log/matching/sft/slm/$slm_model/${noisy}/log/"  \
            --exp_dir "../../../../log/matching/sft/slm/$slm_model/${noisy}/exp/"   \
            --data_name "total" \
            --train_batch_size 16 \
            --eval_batch_size 16 \
            --num_epochs 15.0 \
            --seed 1924 \
            --device "cuda:0"    

        for dir in "${directories[@]}"; do
            echo "Processing directory: $dir with seed: $seed"
    
            python supervised_test.py \
                --model_type "$slm_model" \
                --model_name_or_path "../../../../log/matching/sft/slm/$slm_model/${noisy}/exp/" \
                --data_dir "../../../../log/matching/sft/slm/data/${noisy}/data/"   \
                --log_dir "../../../../log/matching/sft/slm/$slm_model/${noisy}/log/"  \
                --exp_dir "../../../../log/matching/sft/slm/$slm_model/${noisy}/exp/"   \
                --data_name "$dir" \
                --seed 1924 \
                --device "cuda:0"    
        done    
    done
done    