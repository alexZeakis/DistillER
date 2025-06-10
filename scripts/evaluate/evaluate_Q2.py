import pandas as pd
from eval_utils import prepare_plm_file, prepare_ft_file, prepare_umc_file, prepare_df


################### COMPARING PLMS - GENERALIZED ON TEST DATA #################


path = '../../log/matching/plm_generalized/'
total_df_plm_gen = pd.DataFrame()
total_df_plm_gen['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
total_df_plm_gen['LLM-Llama3.1:70b'] = prepare_ft_file('../../log/matching/llm/noisy_llama_70/test_responses/')
total_df_plm_gen['GT'] = prepare_plm_file(path+'ground/')
total_df_plm_gen['Llama3.1:8b'] = prepare_plm_file(path+'noisy_llama_8/')
total_df_plm_gen['Llama3.1:70b'] = prepare_plm_file(path+'noisy_llama_70/')
# total_df_plm_gen['Qwen2.5:7b'] = prepare_plm_file(path+'noisy_qwen_7/')
total_df_plm_gen['Qwen2.5:14b'] = prepare_plm_file(path+'noisy_qwen_14/')
total_df_plm_gen['Qwen2.5:32b'] = prepare_plm_file(path+'noisy_qwen_32/')

total_df_plm_gen = prepare_df(total_df_plm_gen)

latex_code = total_df_plm_gen.to_latex(index=True, escape=False, multirow=False)

