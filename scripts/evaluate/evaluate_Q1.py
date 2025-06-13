import pandas as pd
from eval_utils import prepare_pt_file, prepare_plm_train_file, get_inference_time, \
     get_plm_training_time, get_plm_testing_time, prepare_df

################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../log/matching/annotate/'
total_df_train = pd.DataFrame()
total_df_train['Llama3.1:8b'] = prepare_pt_file(path+'llama_8/partial_responses/')
total_df_train['Llama3.1:70b'] = prepare_pt_file(path+'llama_70/partial_responses/')
# total_df_train['Qwen2.5:7b'] = prepare_pt_file(path+'qwen_7/partial_responses/')
total_df_train['Qwen2.5:14b'] = prepare_pt_file(path+'qwen_14/partial_responses/')
total_df_train['Qwen2.5:32b'] = prepare_pt_file(path+'qwen_32/partial_responses/')
total_df_train['Hybrid-SMiniLM-70b'] = prepare_plm_train_file(path+'sminilm/llama_70/')
total_df_train['Hybrid-RoBERTa-70b'] = prepare_plm_train_file(path+'roberta/llama_70/')

total_df_train = prepare_df(total_df_train)
latex_code = total_df_train.to_latex(index=True, escape=False, multirow=False)


path = '../../log/matching/annotate/'
total_df_train = pd.DataFrame()
total_df_train['Hybrid-SMiniLM-Ground'] = prepare_plm_train_file(path+'sminilm/ground/')
total_df_train['Hybrid-SMiniLM-70b'] = prepare_plm_train_file(path+'sminilm/llama_70/')
total_df_train['Hybrid-RoBERTa-Ground'] = prepare_plm_train_file(path+'roberta/ground/')
total_df_train['Hybrid-RoBERTa-70b'] = prepare_plm_train_file(path+'roberta/llama_70/')

total_df_train = prepare_df(total_df_train)
latex_code_2 = total_df_train.to_latex(index=True, escape=False, multirow=False)

total_df_time = {}
total_df_time['Llama3.1:8b'] = get_inference_time(path+'llama_8/partial_responses/').sum()
total_df_time['Llama3.1:70b'] = get_inference_time(path+'llama_70/partial_responses/').sum()
total_df_time['Qwen2.5:14b'] = get_inference_time(path+'qwen_14/partial_responses/').sum()
total_df_time['Qwen2.5:32b'] = get_inference_time(path+'qwen_32/partial_responses/').sum()
total_df_time['Hybrid-SMiniLM-Inference'] = total_df_time['Llama3.1:70b'] * 0.2
total_df_time['Hybrid-SMiniLM-Training'] = get_plm_training_time(path+'sminilm/llama_70/')
total_df_time['Hybrid-SMiniLM-Testing'] = get_plm_testing_time(path+'sminilm/llama_70/')
total_df_time['Hybrid-SMiniLM-Sum'] = total_df_time['Hybrid-SMiniLM-Inference'] + total_df_time['Hybrid-SMiniLM-Training'] + total_df_time['Hybrid-SMiniLM-Testing']
total_df_time['Hybrid-RoBERTa-Inference'] = total_df_time['Llama3.1:70b'] * 0.2
total_df_time['Hybrid-RoBERTa-Training'] = get_plm_training_time(path+'roberta/llama_70/')
total_df_time['Hybrid-RoBERTa-Testing'] = get_plm_testing_time(path+'roberta/llama_70/')
total_df_time['Hybrid-RoBERTa-Sum'] = total_df_time['Hybrid-RoBERTa-Inference'] + total_df_time['Hybrid-RoBERTa-Training'] + total_df_time['Hybrid-RoBERTa-Testing']
total_df_time = pd.Series(total_df_time).to_frame()

latex_code_time = total_df_time.to_latex(index=True, escape=False, multirow=False)