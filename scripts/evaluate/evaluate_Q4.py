import pandas as pd
from eval_utils import prepare_plm_file_cardinality, prepare_plm_file, prepare_ft_file, prepare_umc_file, prepare_df


################### COMPARING PLMS - GENERALIZED ON TEST DATA #################


path = '../../log/matching/plm/'
df = pd.DataFrame()

df['SMiniLM-GT'] = prepare_plm_file_cardinality(path+'sminilm/ground/')
df['SMiniLM-Llama3.1:8b'] = prepare_plm_file_cardinality(path+'sminilm/llama_8/')
df['SMiniLM-Llama3.1:70b'] = prepare_plm_file_cardinality(path+'sminilm/llama_70/')
df['SMiniLM-Qwen2.5:14b'] = prepare_plm_file_cardinality(path+'sminilm/qwen_14/')
df['SMiniLM-Qwen2.5:32b'] = prepare_plm_file_cardinality(path+'sminilm/qwen_32/')

df['RoBERTa-GT'] = prepare_plm_file_cardinality(path+'roberta/ground/')
df['RoBERTa-Llama3.1:8b'] = prepare_plm_file_cardinality(path+'roberta/llama_8/')
df['RoBERTa-Llama3.1:70b'] = prepare_plm_file_cardinality(path+'roberta/llama_70/')
df['RoBERTa-Qwen2.5:14b'] = prepare_plm_file_cardinality(path+'roberta/qwen_14/')
df['RoBERTa-Qwen2.5:32b'] = prepare_plm_file_cardinality(path+'roberta/qwen_32/')

# df = prepare_df(df)
df = df.T

latex_code = df.to_latex(index=True, escape=False, multirow=False)


path2 = '../../log/matching/disambiguation/'
df = pd.DataFrame()
df['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
df['LLM-Llama3.1:70b'] = prepare_ft_file('../../log/matching/llm/llama_70/test_responses/')

df['SMiniLM-GT'] = prepare_plm_file(path+'sminilm/ground/')
df['SMiniLM-GT-UMC'] = prepare_plm_file(path2+'umc/sminilm/ground/', 'final/')
df['SMiniLM-GT-SEL'] = prepare_plm_file(path2+'select/sminilm/ground/', 'final/')
df['SMiniLM-Llama3.1:70b'] = prepare_plm_file(path+'sminilm/llama_70/')
df['SMiniLM-Llama3.1:70b-UMC'] = prepare_plm_file(path2+'umc/sminilm/llama_70/', 'final/')
df['SMiniLM-Llama3.1:70b-SEL'] = prepare_plm_file(path2+'select/sminilm/llama_70/', 'final/')

df['RoBERTa-GT'] = prepare_plm_file(path+'roberta/ground/')
df['RoBERTa-GT-UMC'] = prepare_plm_file(path2+'umc/roberta/ground/', 'final/')
df['RoBERTa-GT-SEL'] = prepare_plm_file(path2+'select/roberta/ground/', 'final/')
df['RoBERTa-Llama3.1:70b'] = prepare_plm_file(path+'roberta/llama_70/')
df['RoBERTa-Llama3.1:70b-UMC'] = prepare_plm_file(path2+'umc/roberta/llama_70/', 'final/')
df['RoBERTa-Llama3.1:70b-SEL'] = prepare_plm_file(path2+'select/roberta/llama_70/', 'final/')

df = prepare_df(df)

latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)