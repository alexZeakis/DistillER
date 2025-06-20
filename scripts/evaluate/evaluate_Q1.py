import pandas as pd
from eval_utils import prepare_pt_file, prepare_plm_train_file, get_inference_time, \
     get_plm_training_time, get_plm_testing_time, prepare_df

################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../log/matching/annotate/'
df = pd.DataFrame()
df['Llama3.1:8b'] = prepare_pt_file(path+'llama_8/partial_responses/')
df['Llama3.1:70b'] = prepare_pt_file(path+'llama_70/partial_responses/')
# df['Qwen2.5:7b'] = prepare_pt_file(path+'qwen_7/partial_responses/')
df['Qwen2.5:14b'] = prepare_pt_file(path+'qwen_14/partial_responses/')
df['Qwen2.5:32b'] = prepare_pt_file(path+'qwen_32/partial_responses/')
df['Hybrid-SMiniLM-70b'] = prepare_plm_train_file(path+'sminilm/llama_70/')
df['Hybrid-RoBERTa-70b'] = prepare_plm_train_file(path+'roberta/llama_70/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)


path = '../../log/matching/annotate/'
df = pd.DataFrame()
df['Hybrid-SMiniLM-Ground'] = prepare_plm_train_file(path+'sminilm/ground/')
df['Hybrid-SMiniLM-70b'] = prepare_plm_train_file(path+'sminilm/llama_70/')
df['Hybrid-RoBERTa-Ground'] = prepare_plm_train_file(path+'roberta/ground/')
df['Hybrid-RoBERTa-70b'] = prepare_plm_train_file(path+'roberta/llama_70/')

df = prepare_df(df)
latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)

df = {}
df['Llama3.1:8b'] = get_inference_time(path+'llama_8/partial_responses/').sum()
df['Llama3.1:70b'] = get_inference_time(path+'llama_70/partial_responses/').sum()
df['Qwen2.5:14b'] = get_inference_time(path+'qwen_14/partial_responses/').sum()
df['Qwen2.5:32b'] = get_inference_time(path+'qwen_32/partial_responses/').sum()
df['Hybrid-SMiniLM-Inference'] = df['Llama3.1:70b'] * 0.2
df['Hybrid-SMiniLM-Training'] = get_plm_training_time(path+'sminilm/llama_70/')
df['Hybrid-SMiniLM-Testing'] = get_plm_testing_time(path+'sminilm/llama_70/')
df['Hybrid-SMiniLM-Sum'] = df['Hybrid-SMiniLM-Inference'] + df['Hybrid-SMiniLM-Training'] + df['Hybrid-SMiniLM-Testing']
df['Hybrid-RoBERTa-Inference'] = df['Llama3.1:70b'] * 0.2
df['Hybrid-RoBERTa-Training'] = get_plm_training_time(path+'roberta/llama_70/')
df['Hybrid-RoBERTa-Testing'] = get_plm_testing_time(path+'roberta/llama_70/')
df['Hybrid-RoBERTa-Sum'] = df['Hybrid-RoBERTa-Inference'] + df['Hybrid-RoBERTa-Training'] + df['Hybrid-RoBERTa-Testing']
df = pd.Series(df).to_frame()

latex_code_time = df.to_latex(index=True, escape=False, multirow=False)