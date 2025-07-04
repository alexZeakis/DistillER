import pandas as pd
from eval_utils import prepare_plm_file, prepare_ft_file, prepare_umc_file, prepare_df
from time import time

################### COMPARING PLMS - GENERALIZED ON TEST DATA #################

t1 = time()
path = '../../log/matching/slm/'
df = pd.DataFrame()
df['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
# df['LLM-Llama3.1:70b'] = prepare_ft_file('../../log/matching/llm/llama_70/test_responses/')

df['SMiniLM-GT'] = prepare_plm_file(path+'sminilm/ground/')
# df['SMiniLM-GT2'] = prepare_plm_file(path+'sminilm/ground_2/')
# # df['SMiniLM-Llama3.1:8b'] = prepare_plm_file(path+'sminilm/llama_8/')
# # df['SMiniLM-Llama3.1:70b'] = prepare_plm_file(path+'sminilm/llama_70/')
# # df['SMiniLM-Qwen2.5:14b'] = prepare_plm_file(path+'sminilm/qwen_14/')
df['SMiniLM-Qwen2.5:32b'] = prepare_plm_file(path+'sminilm/qwen_32/')
df['SMiniLM-Hybrid'] = prepare_plm_file(path+'sminilm/roberta/qwen_32/')

df['RoBERTa-GT'] = prepare_plm_file(path+'roberta/ground/')
# df['RoBERTa-GT2'] = prepare_plm_file(path+'roberta/ground_2/')
# # df['RoBERTa-Llama3.1:8b'] = prepare_plm_file(path+'roberta/llama_8/')
# # df['RoBERTa-Llama3.1:70b'] = prepare_plm_file(path+'roberta/llama_70/')
# # df['RoBERTa-Qwen2.5:14b'] = prepare_plm_file(path+'roberta/qwen_14/')
df['RoBERTa-Qwen2.5:32b'] = prepare_plm_file(path+'roberta/roberta/qwen_32/')
df['RoBERTa-Hybrid'] = prepare_plm_file(path+'roberta/roberta/qwen_32/')

df = prepare_df(df)

latex_code = df.to_latex(index=True, escape=False, multirow=False)

t2 = time()
print('Time elapsed {:.2f}'.format(t2-t1))