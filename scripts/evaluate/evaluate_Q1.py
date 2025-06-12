import pandas as pd
from eval_utils import prepare_pt_file, prepare_plm_train_file, get_inference_time, \
     get_plm_training_time, get_plm_testing_time, prepare_df

################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../log/matching/llm/'
path2 = '../../log/matching/hybrid/'
total_df_train = pd.DataFrame()
total_df_train['Llama3.1:8b'] = prepare_pt_file(path+'noisy_llama_8/partial_responses/')
total_df_train['Llama3.1:70b'] = prepare_pt_file(path+'noisy_llama_70/partial_responses/')
# total_df_train['Qwen2.5:7b'] = prepare_pt_file(path+'noisy_qwen_7/partial_responses/')
total_df_train['Qwen2.5:14b'] = prepare_pt_file(path+'noisy_qwen_14/partial_responses/')
total_df_train['Qwen2.5:32b'] = prepare_pt_file(path+'noisy_qwen_32/partial_responses/')

# total_df_train['Qwen2.5:7b'] = prepare_ft_file(path2+'/noisy_qwen_7/partial_responses/')
total_df_train['PLM-Ground'] = prepare_plm_train_file(path2+'ground/')
total_df_train['PLM-70b'] = prepare_plm_train_file(path2+'noisy_llama_70/')

total_df_train = prepare_df(total_df_train)

latex_code = total_df_train.to_latex(index=True, escape=False, multirow=False)


total_df_time = {}
total_df_time['Llama3.1:8b'] = get_inference_time(path+'noisy_llama_8/partial_responses/').sum()
total_df_time['Llama3.1:70b'] = get_inference_time(path+'noisy_llama_70/partial_responses/').sum()
total_df_time['Qwen2.5:14b'] = get_inference_time(path+'noisy_qwen_14/partial_responses/').sum()
total_df_time['Qwen2.5:32b'] = get_inference_time(path+'noisy_qwen_32/partial_responses/').sum()
total_df_time['Hybrid-Inference'] = total_df_time['Llama3.1:70b'] * 0.2
total_df_time['Hybrid-Training'] = get_plm_training_time(path2+'noisy_llama_70/')
total_df_time['Hybrid-Testing'] = get_plm_testing_time(path2+'noisy_llama_70/')
total_df_time['Hybrid-Sum'] = total_df_time['Hybrid-Inference'] + total_df_time['Hybrid-Training'] + total_df_time['Hybrid-Testing']
total_df_time = pd.Series(total_df_time).to_frame()

latex_code = total_df_time.to_latex(index=True, escape=False, multirow=False)