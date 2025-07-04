import logging
import os
from datetime import datetime
import pandas as pd

from hybrid_utils import initialize_gpu_seed, load_data, DataType, \
                             DeepMatcherProcessor, setup_logging, \
                             read_arguments_train, load_model, \
                             get_scores_from_json, get_scores_from_ft_json, \
                             get_prompts_for_json

setup_logging()


if __name__ == "__main__":
    args = read_arguments_train()  # same arg parser used in training
    
    final_df = pd.DataFrame()
    
    slm_predictions = pd.read_csv(args.slm_predictions, index_col=0)
    
    if 'llm' in args.llm_predictions:
        # llm_predictions, llm_ground_results = get_scores_from_ft_json(args.llm_predictions)
        llm_predictions = get_scores_from_ft_json(args.llm_predictions)
    else:
        # llm_predictions, llm_ground_results = get_scores_from_json(args.llm_predictions)
        llm_predictions = get_scores_from_json(args.llm_predictions)

    options = get_prompts_for_json(args.llm_prompts)

    model_dir = args.model_name_or_path
    device, n_gpu = initialize_gpu_seed(args.device, args.seed)

    processor = DeepMatcherProcessor()

    model, tokenizer = load_model(model_dir, device)
    logging.info("Model loaded from {}".format(model_dir))

    test_examples = processor.get_test_examples(args.data_path)
    logging.info("Loaded {} test examples".format(len(test_examples)))

    exceeded = load_data(test_examples, tokenizer, DataType.TEST)

    no_yes, no_no = 0, 0
    for left_id, candidates in exceeded.items():
        total_exceeded = sum(candidates.values())
        temp_df = slm_predictions.loc[slm_predictions.left_ID==left_id].copy()
        if total_exceeded == 0: #SLM appropriate
            no_no += 1
        else: #LLM appropriate
            llm_predict = llm_predictions.get(left_id, -1) # in case llm also missed it
            if llm_predict > 0 and llm_predict-1 < len(options[left_id]): # covers -1 and 0 cases
                llm_predict = options[left_id][llm_predict-1]
            else:
                llm_predict = -1
            temp_df.predictions = (temp_df.right_ID == llm_predict)*1
            no_yes += 1
            
        final_df = pd.concat([final_df, temp_df])
        
    # print('Total: {}, Yes: {}, No: {}'.format(len(exceeded), no_yes, no_no))


    # Save predictions
    path2 = os.path.join(args.log_dir, args.data_name + '_predictions.csv')
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    final_df.to_csv(path2)