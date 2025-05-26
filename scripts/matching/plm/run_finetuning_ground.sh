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
        --out_dir "../../../log/matching/plm/ground/data/$dir/"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
        --seed $seed \
        --serialization "DITTO" \
        --percentage 0.8 \
        --mode "train" 
        
        python transform_data_plm.py \
        --dataset "$dir" \
        --out_dir "../../../log/matching/plm/ground/data/$dir/"  \
        --in_dir "../../../data/ccer/cleaned/original/" \
        --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
        --seed $seed \
        --serialization "DITTO" \
        --mode "test" 
        
    done
    
    python supervised_main.py \
     --model_type "sminilm" \
     --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
     --data_dir "../../../log/matching/plm/ground/data/"   \
     --log_dir "../../../log/matching/plm/ground/log/"  \
     --exp_dir "../../../log/matching/plm/ground/exp/"   \
     --data_name "$dir" \
     --train_batch_size 16 \
     --eval_batch_size 16 \
     --num_epochs 15.0 \
     --seed 1924 \
     --device "cuda:0"

    python umc.py \
      --predictions_file "../../../log/matching/plm/ground/log/${dir}_predictions.csv" \
      --input_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
      --out_file "../../../log/matching/plm/ground/umc/${dir}.csv"

     
done