# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

# Seeds
seeds=(1924)
noisy=noisy_llama_70

for dir in "${directories[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Processing directory: $dir with seed: $seed"

        python ../plm/transform_data_plm.py \
            --dataset "$dir" \
            --out_dir "../../../log/matching/hybrid/${noisy}/data/$dir/"  \
            --in_dir "../../../data/ccer/cleaned/original/" \
            --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
            --seed $seed \
            --serialization "DITTO" \
            --percentage 0.8 \
            --mode "train" \
            --response_file "../../../log/matching/finetuning/${noisy}/partial_responses/${dir}_${seed}_responses.json"
    done
done

python merge_data.py \
    --in_dir "../../../log/matching/hybrid/${noisy}/data" \
    --out_dir "../../../log/matching/hybrid/${noisy}/data/total/" \
    --percentage 0.8


python ../plm/supervised_main.py \
    --model_type "sminilm" \
    --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
    --data_dir "../../../log/matching/hybrid/${noisy}/data/"   \
    --log_dir "../../../log/matching/hybrid/${noisy}/log/"  \
    --exp_dir "../../../log/matching/hybrid/${noisy}/exp/"   \
    --data_name "total" \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_epochs 15.0 \
    --seed 1924 \
    --device "cuda:0"    


for dir in "${directories[@]}"; do
    echo "Processing directory: $dir with seed: $seed"

    python ../build_prompt.py \
    --dataset "$dir" \
    --out_file "../../../log/matching/hybrid/${noisy}/partial/${dir}_1924.json"  \
    --in_dir "../../../data/ccer/cleaned/original/" \
    --sample_file "../../../data/ccer/cleaned/fine_tuning/train/$dir.csv" \
    --seed 1924 \
    --serialization "DITTO" \
    --task_description "EXPLAIN"

    python ../embed_noisy.py \
    --prompts "../../../log/matching/hybrid/${noisy}/partial/${dir}_1924.json"  \
    --labels "../../../log/matching/hybrid/${noisy}/log/total_predictions.csv" \
    --out_file "../../../log/matching/hybrid/${noisy}/partial_noisy/${dir}_1924.json" 
done    

<<xom
input_files=(../../../log/matching/hybrid/${noisy}/partial_noisy/*.json)
python ../fine_tuning_data.py \
   --input_files "${input_files[@]}" \
   --out_file "../../../log/matching/hybrid/${noisy}/train.json" \
   --mode "train" \
   --field "noise_answer"

python ../fine_tuning_train.py \
   --input_file "../../../log/matching/hybrid/${noisy}/train.json" \
   --log_file "../../../log/matching/hybrid/${noisy}/train_log.json" \
   --out_dir "../../../log/matching/hybrid/${noisy}/llama31_gt" \
   --out_name "llama31_gt" \
   --model "llama3.1"

for dir in "${directories[@]}"; do
    echo "Processing directory: $dir with seed: 1924"
   
    python ../build_prompt.py \
    --dataset "$dir" \
    --out_file "../../../log/matching/hybrid/${noisy}/test/${dir}_1924.json"  \
    --in_dir "../../../data/ccer/cleaned/original/" \
    --sample_file "../../../data/ccer/cleaned/fine_tuning/test/$dir.csv" \
    --seed 1924 \
    --serialization "DITTO" \
    --task_description "EXPLAIN"
    
    python ../fine_tuning_data.py \
       --input_files "../../../log/matching/hybrid/${noisy}/test/${dir}_1924.json" \
       --out_file "../../../log/matching/hybrid/${noisy}/test/${dir}_total.json" \
       --mode "test"
    
    python ../fine_tuning_test.py \
       --dataset "$dir" \
       --model_path "../../../log/matching/hybrid/${noisy}/llama31_gt" \
       --input_file "../../../log/matching/hybrid/${noisy}/test/${dir}_total.json" \
       --out_file "../../../log/matching/hybrid/${noisy}/test_responses/${dir}_responses.json"
done
xom