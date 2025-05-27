# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Seeds
seeds=(1924)

for dir in "${directories[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Processing directory: $dir with seed: $seed"

        python transform_data_plm.py \
            --dataset "$dir" \
            --out_dir "../../../log/matching/plm_generalized/ground/data/$dir/"  \
            --in_dir "../../../data/ccer/cleaned/original/" \
            --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
            --seed $seed \
            --serialization "DITTO" \
            --percentage 0.8 \
            --mode "train" \

        python transform_data_plm.py \
            --dataset "$dir" \
            --out_dir "../../../log/matching/plm_generalized/ground/data/$dir/"  \
            --in_dir "../../../data/ccer/cleaned/original/" \
            --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
            --seed $seed \
            --serialization "DITTO" \
            --mode "test" 
    done
done

python merge_data.py \
    --in_dir "../../../log/matching/plm_generalized/ground/data" \
    --out_dir "../../../log/matching/plm_generalized/ground/data/total/"

python supervised_train.py \
    --model_type "sminilm" \
    --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
    --data_dir "../../../log/matching/plm_generalized/ground/data/"   \
    --log_dir "../../../log/matching/plm_generalized/ground/log/"  \
    --exp_dir "../../../log/matching/plm_generalized/ground/exp/"   \
    --data_name "total" \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_epochs 15.0 \
    --seed 1924 \
    --device "cuda:0"    

for dir in "${directories[@]}"; do
    echo "Processing directory: $dir with seed: $seed"

    python supervised_test.py \
        --model_type "sminilm" \
        --model_name_or_path "../../../log/matching/plm_generalized/ground/exp/" \
        --data_dir "../../../log/matching/plm_generalized/ground/data/"   \
        --log_dir "../../../log/matching/plm_generalized/ground/log/"  \
        --exp_dir "../../../log/matching/plm_generalized/ground/exp/"   \
        --data_name "$dir" \
        --seed 1924 \
        --device "cuda:0"    
done    
