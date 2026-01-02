import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import get_inference_time, get_plm_time, prepare_df, prepare_train_file, prepare_any

################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../../log/matching/annotate/llm/'
path2 = '../../../log/matching/annotate/slm/llm/'
gt_path = '../../../log/matching/annotate/llm/ground/partial/'

latex_code = []
for metric in ['precision', 'recall']:
    df = pd.DataFrame()
    df['Llama:70b'] = prepare_train_file(path+'llama_70/', gt_path, metric=metric)
    df['Qwen:32b'] = prepare_train_file(path+'qwen_32/', gt_path, metric=metric)
    df['Llama:8b'] = prepare_train_file(path+'llama_8/', gt_path, metric=metric)
    df['Qwen:14b'] = prepare_train_file(path+'qwen_14/', gt_path, metric=metric)
    df['SMiniLM-32'] = prepare_train_file(path2+'sminilm/qwen_32/', gt_path, metric=metric)
    df['RoBERTa-32'] = prepare_train_file(path2+'roberta/qwen_32/', gt_path, metric=metric)
    df['SMiniLM-70'] = prepare_train_file(path2+'sminilm/llama_70/', gt_path, metric=metric)
    df['RoBERTa-70'] = prepare_train_file(path2+'roberta/llama_70/', gt_path, metric=metric)
    df['Multi-Teacher'] = prepare_train_file(path+'multi_llm/', gt_path, metric=metric)
    
    df['Any'] = prepare_any([path+'llama_70/', path+'qwen_32/', path+'llama_8/', path+'qwen_14/'], 
                              gt_path, metric=metric)
    df = prepare_df(df)
    
    # print(df)
    latex_code.append(df.to_latex(index=True, escape=False, multirow=False))