# List of directories
directories=("D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9")
#directories=("D3")


# slm_annotator levels
#slm_annotators=("ground" "qwen_32" "roberta/qwen_32")
#slm_annotators=("roberta/qwen_32")
slm_annotators=("qwen_32")
slm_models=("roberta" "sminilm")

llm_models=("llm/qwen_32" "baselines/pretrained")
#llm_models=("baselines/pretrained")

for slm_model in "${slm_models[@]}"; do
    # Loop over each slm_annotator level
    echo -e "=== Processing: $slm_model ==="
    for slm_annotator in "${slm_annotators[@]}"; do
        echo -e "\t=== Processing: $slm_annotator ==="
    
        for llm_model in "${llm_models[@]}"; do    
            echo -e "\t\t===Processing: $llm_model"

            for dir in "${directories[@]}"; do            
                echo -e "\t\t\tProcessing directory: $dir"
                
                if [ "$llm_model" = "baselines/pretrained" ]; then
                    llm_path="baselines/pretrained"
                    prompts_path="baselines/pretrained/${dir}.json"
                else
                    llm_path="$llm_model/test_responses"
                    #prompts_path="$llm_model/test/${dir}_total.json"
                    prompts_path="baselines/pretrained/${dir}.json"
                fi    
            
                python hybrid.py \
                    --model_type "${slm_model}" \
                    --model_name_or_path "../../../log/matching/slm/${slm_model}/${slm_annotator}/exp/" \
                    --data_dir "../../../log/matching/slm/data/${slm_annotator}/data/"   \
                    --log_dir "../../../log/matching/hybrid/${slm_model}/${slm_annotator}/${llm_model}/log/"  \
                    --slm_predictions "../../../log/matching/disambiguation/select/${slm_model}/${slm_annotator}/final/${dir}.csv" \
                    --llm_predictions "../../../log/matching/${llm_path}/${dir}_responses.json" \
                    --llm_prompts "../../../log/matching/"$prompts_path \
                    --data_name "$dir" \
                    --seed 1924 \
                    --device "cuda:0"    
                    
            done
            #break
        done
        #break
    done
    #break
done    

#--slm_predictions "../../../log/matching/slm/${slm_model}/${slm_annotator}/log/${dir}_predictions.csv" \