# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Seeds
seeds=(1924)

for dir in "${directories[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Processing directory: $dir with seed: $seed"

        python transform_data_plm.py \
            --dataset "$dir" \
            --out_dir "../../../log/matching/plm/data/ground/data/$dir/"  \
            --in_dir "../../../data/ccer/cleaned/original/" \
            --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
            --seed $seed \
            --serialization "DITTO" \
            --percentage 0.8 \
            --mode "train" \

        python transform_data_plm.py \
            --dataset "$dir" \
            --out_dir "../../../log/matching/plm/data/ground/data/$dir/"  \
            --in_dir "../../../data/ccer/cleaned/original/" \
            --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
            --seed $seed \
            --serialization "DITTO" \
            --mode "test" 
    done
done

python merge_data.py \
    --in_dir "../../../log/matching/plm/data/ground/data" \
    --out_dir "../../../log/matching/plm/data/ground/data/total/"