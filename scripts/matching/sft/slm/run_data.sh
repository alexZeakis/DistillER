# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Seeds
seeds=(1924)

noisy_levels=("ground" "qwen_32" "roberta/qwen_32")

# Loop over each noise level
for noisy in "${noisy_levels[@]}"; do
    echo "=== Processing for noise level: $noisy ==="

    for dir in "${directories[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Processing directory: $dir with seed: $seed"

            if [ "$noisy" = "ground" ]; then
                extra_args=""
            elif [[ "$noisy" == llama* || "$noisy" == qwen* ]]; then
                extra_args="--response_file ../../../../log/matching/annotate/llm/${noisy}/partial_noisy/${dir}_${seed}.json"
            elif [[ "$noisy" == sminilm* || "$noisy" == roberta* ]]; then
                extra_args="--response_file ../../../../log/matching/annotate/slm/llm/${noisy}/partial_noisy/${dir}_${seed}.json"
            else
                echo "Unknown noisy value: $noisy"
                exit 1
            fi            

            # Train
            python transform_data_plm.py \
                --dataset "$dir" \
                --out_dir "../../../../log/matching/sft/slm/data/${noisy}/data/$dir/" \
                --in_dir "../../../../data/ccer/cleaned/original/" \
                --sample_file "../../../../data/ccer/cleaned/fine_tuning/blocking_max/train/$dir.csv" \
                --seed $seed \
                --serialization "DITTO" \
                --percentage 0.8 \
                --mode "train" \
                $extra_args

            # Test
            python transform_data_plm.py \
                --dataset "$dir" \
                --out_dir "../../../../log/matching/sft/slm/data/${noisy}/data/$dir/" \
                --in_dir "../../../../data/ccer/cleaned/original/" \
                --sample_file "../../../../data/ccer/cleaned/fine_tuning/blocking_max/test/$dir.csv" \
                --seed $seed \
                --serialization "DITTO" \
                --mode "test"
        done
    done


    # Merge
    python merge_data.py \
        --in_dir "../../../../log/matching/sft/slm/data/${noisy}/data" \
        --out_dir "../../../../log/matching/sft/slm/data/${noisy}/data/total/"
        
done
