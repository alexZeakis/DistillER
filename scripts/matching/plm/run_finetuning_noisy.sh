# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

#seeds=(1924 1913 1941 1830 1204)
seeds=(1924)

# Loop over each seed and directory, and run the Python scripts
for dir in "${directories[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Processing directory: $dir with seed: $seed"
       
        python transform_data_plm.py \
        --dataset "$dir" \
        --out_dir "../../../log/matching/plm/noisy_6/data/$dir/"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
        --seed $seed \
        --serialization "DITTO" \
        --percentage 0.8 \
        --mode "train" \
        --response_file "../../../log/matching/finetuning/noisy_6/partial_responses/${dir}_${seed}_responses.json"
        
        python transform_data_plm.py \
        --dataset "$dir" \
        --out_dir "../../../log/matching/plm/noisy_6/data/$dir/"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
        --seed $seed \
        --serialization "DITTO" \
        --mode "test" 
        
    done
    
    python supervised_main.py \
     --model_type "sminilm" \
     --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
     --data_dir "../../../log/matching/plm/noisy_6/data/"   \
     --log_dir "../../../log/matching/plm/noisy_6/log/"  \
     --exp_dir "../../../log/matching/plm/noisy_6/exp/"   \
     --data_name "$dir" \
     --train_batch_size 16 \
     --eval_batch_size 16 \
     --num_epochs 15.0 \
     --seed 1924 \
     --device "cuda:0"

done