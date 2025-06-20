import pandas as pd
from eval_utils import prepare_plm_file, prepare_ft_file, prepare_umc_file, prepare_df


################### COMPARING PLMS - GENERALIZED ON TEST DATA #################


path = '../../log/matching/plm/'
df = pd.DataFrame()
df['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
df['LLM-Llama3.1:70b'] = prepare_ft_file('../../log/matching/llm/llama_70/test_responses/')

df['SMiniLM-GT'] = prepare_plm_file(path+'sminilm/ground/')
df['SMiniLM-Llama3.1:8b'] = prepare_plm_file(path+'sminilm/llama_8/')
df['SMiniLM-Llama3.1:70b'] = prepare_plm_file(path+'sminilm/llama_70/')
df['SMiniLM-Qwen2.5:14b'] = prepare_plm_file(path+'sminilm/qwen_14/')
df['SMiniLM-Qwen2.5:32b'] = prepare_plm_file(path+'sminilm/qwen_32/')

df['RoBERTa-GT'] = prepare_plm_file(path+'roberta/ground/')
df['RoBERTa-Llama3.1:8b'] = prepare_plm_file(path+'roberta/llama_8/')
df['RoBERTa-Llama3.1:70b'] = prepare_plm_file(path+'roberta/llama_70/')
df['RoBERTa-Qwen2.5:14b'] = prepare_plm_file(path+'roberta/qwen_14/')
df['RoBERTa-Qwen2.5:32b'] = prepare_plm_file(path+'roberta/qwen_32/')

df = prepare_df(df)

latex_code = df.to_latex(index=True, escape=False, multirow=False)

