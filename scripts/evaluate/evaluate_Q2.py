import pandas as pd
from eval_utils import prepare_umc_file, prepare_pt_file, prepare_ft_file, prepare_df


################### COMPARING LLMS ON TEST DATA #########################

total_df_test = pd.DataFrame()

path = '../../log/matching/llm/'
total_df_test['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
total_df_test['Pretrained'] = prepare_pt_file('../../log/matching/baselines/pretrained/')    
total_df_test['GT'] = prepare_ft_file(path+'ground/test_responses/')
total_df_test['Llama3.1:8b'] = prepare_ft_file(path+'noisy_llama_8/test_responses/')
total_df_test['Llama3.1:70b'] = prepare_ft_file(path+'noisy_llama_70/test_responses/')
# total_df_test['Qwen2.5:7b'] = prepare_ft_file(path+'llm/noisy_qwen_7/test_responses/')
total_df_test['Qwen2.5:14b'] = prepare_ft_file(path+'noisy_qwen_14/test_responses/')
total_df_test['Qwen2.5:32b'] = prepare_ft_file(path+'noisy_qwen_32/test_responses/')

total_df_test['H-SMiniLM'] = prepare_ft_file(path+'sminilm/noisy_llama_70/test_responses/')
total_df_test['H-RoBERTa'] = prepare_ft_file(path+'roberta/noisy_llama_70/test_responses/')

total_df_test = prepare_df(total_df_test)

# total_df[['UMC', 'Pretrained', 'GT', 'Llama3.1:8b', 'Qwen2.5:32b']].plot.line()
# create_plot_comparison(total_df_test)

