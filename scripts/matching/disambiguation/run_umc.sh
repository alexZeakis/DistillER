# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")

noisy_levels=("ground" "qwen_32" "roberta/qwen_32")

models=("roberta" "sminilm")

# Loop over each noisy level
for model in "${models[@]}"; do
    for noisy in "${noisy_levels[@]}"; do
        for dir in "${directories[@]}"; do
            echo $model " " $noisy " " $dir
            
            python clean_edges.py \
              --predictions_file "../../../log/matching/slm/$model/${noisy}/log/${dir}_predictions.csv" \
              --input_file "../../../data/ccer/cleaned/fine_tuning/blocking/test/$dir.csv" \
              --out_file "../../../log/matching/disambiguation/cleaned_edges/$model/${noisy}/${dir}.csv"

            python umc.py \
             --in_file "../../../log/matching/disambiguation/cleaned_edges/$model/${noisy}/${dir}.csv" \
             --out_file "../../../log/matching/disambiguation/umc/$model/${noisy}/umc/${dir}.csv"

            python merge_umc.py \
               --input_file "../../../log/matching/slm/$model/${noisy}/log/${dir}_predictions.csv" \
               --cleaned_file "../../../log/matching/disambiguation/umc/$model/${noisy}/umc/${dir}.csv" \
               --out_file "../../../log/matching/disambiguation/umc/$model/${noisy}/final/${dir}.csv"
        done
    done
done
