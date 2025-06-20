# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Seeds
seeds=(1924)

# Noisy levels
noisy_levels=("llama_8" "llama_70" "qwen_14" "qwen_32")

# Loop over each noisy level
for noisy in "${noisy_levels[@]}"; do
    echo "=== Processing for noise level: $noisy ==="
    
    for dir in "${directories[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Processing directory: $dir with seed: $seed"

            python transform_data_plm.py \
                --dataset "$dir" \
                --out_dir "../../../log/matching/plm/data/${noisy}/data/$dir/"  \
                --in_dir "../../../data/ccer/cleaned/original/" \
                --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
                --seed $seed \
                --serialization "DITTO" \
                --percentage 0.8 \
                --mode "train" \
                --response_file "../../../log/matching/annotate/${noisy}/partial_responses/${dir}_${seed}_responses.json"

            python transform_data_plm.py \
                --dataset "$dir" \
                --out_dir "../../../log/matching/plm/data/${noisy}/data/$dir/"  \
                --in_dir "../../../data/ccer/cleaned/original/" \
                --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
                --seed $seed \
                --serialization "DITTO" \
                --mode "test" 
        done
    done
    
    python merge_data.py \
        --in_dir "../../../log/matching/plm/data/${noisy}/data" \
        --out_dir "../../../log/matching/plm/data/${noisy}/data/total/"

done