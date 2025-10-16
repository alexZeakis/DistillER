import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import get_inference_time, get_plm_training_time, \
    get_plm_testing_time, prepare_df, prepare_train_file

################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../../log/matching/annotate/blocking/'
path2 = '../../../log/matching/annotate/slm/llm/'
gt_path = '../../../log/matching/annotate/blocking/ground/partial/'
df = pd.DataFrame()
df['Llama3.1:70b'] = prepare_train_file(path+'llama_70/', gt_path)
df['Qwen2.5:32b'] = prepare_train_file(path+'qwen_32/', gt_path)
df['SMiniLM-32'] = prepare_train_file(path2+'sminilm/qwen_32/', gt_path)
df['RoBERTa-32'] = prepare_train_file(path2+'roberta/qwen_32/', gt_path)

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)



df = {}
df['Llama3.1:70b'] = get_inference_time(path+'llama_70/partial_responses/').sum()
df['Qwen2.5:32b'] = get_inference_time(path+'qwen_32/partial_responses/').sum()
df['Hybrid-SMiniLM-Inference'] = df['Qwen2.5:32b'] * 0.2
df['Hybrid-SMiniLM-Training'] = get_plm_training_time(path2+'sminilm/qwen_32/')
df['Hybrid-SMiniLM-Testing'] = get_plm_testing_time(path2+'sminilm/llama_70/')
df['Hybrid-SMiniLM-Sum'] = df['Hybrid-SMiniLM-Inference'] + df['Hybrid-SMiniLM-Training'] + df['Hybrid-SMiniLM-Testing']
df['Hybrid-RoBERTa-Inference'] = df['Qwen2.5:32b'] * 0.2
df['Hybrid-RoBERTa-Training'] = get_plm_training_time(path2+'roberta/qwen_32/')
df['Hybrid-RoBERTa-Testing'] = get_plm_testing_time(path2+'roberta/qwen_32/')
df['Hybrid-RoBERTa-Sum'] = df['Hybrid-RoBERTa-Inference'] + df['Hybrid-RoBERTa-Training'] + df['Hybrid-RoBERTa-Testing']
df = pd.Series(df).to_frame()

latex_code_time = df.to_latex(index=True, escape=False, multirow=False)