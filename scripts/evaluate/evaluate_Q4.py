import pandas as pd
from eval_utils import prepare_plm_file_cardinality, prepare_plm_file, prepare_ft_file, prepare_umc_file, prepare_df


# ################### COMPARING PLMS - GENERALIZED ON TEST DATA #################

path = '../../log/matching/slm/'
df = pd.DataFrame()

df['SMiniLM-GT'] = prepare_plm_file_cardinality(path+'sminilm/ground/')
df['SMiniLM-LLM'] = prepare_plm_file_cardinality(path+'sminilm/qwen_32/')
df['SMiniLM-SLM'] = prepare_plm_file_cardinality(path+'sminilm/roberta/qwen_32/')
df['RoBERTa-GT'] = prepare_plm_file_cardinality(path+'roberta/ground/')
df['RoBERTa-LLM'] = prepare_plm_file_cardinality(path+'roberta/qwen_32/')
df['RoBERTa-SLM'] = prepare_plm_file_cardinality(path+'roberta/roberta/qwen_32/')

# df = prepare_df(df)
df = df.T

latex_code = df.to_latex(index=True, escape=False, multirow=False)


path = '../../log/matching/slm/'
path2 = '../../log/matching/disambiguation/'
df = pd.DataFrame()

df['SMiniLM-GT'] = prepare_plm_file(path+'sminilm/ground/')
df['SMiniLM-GT-UMC'] = prepare_plm_file(path2+'umc/sminilm/ground/', 'final/')
df['SMiniLM-GT-SEL'] = prepare_plm_file(path2+'select/sminilm/ground/', 'final/')
df['SMiniLM-LLM'] = prepare_plm_file(path+'sminilm/qwen_32/')
df['SMiniLM-LLM-UMC'] = prepare_plm_file(path2+'umc/sminilm/qwen_32/', 'final/')
df['SMiniLM-LLM-SEL'] = prepare_plm_file(path2+'select/sminilm/qwen_32/', 'final/')
df['SMiniLM-SLM'] = prepare_plm_file(path+'sminilm/roberta/qwen_32/')
df['SMiniLM-SLM-UMC'] = prepare_plm_file(path2+'umc/sminilm/roberta/qwen_32/', 'final/')
df['SMiniLM-SLM-SEL'] = prepare_plm_file(path2+'select/sminilm/roberta/qwen_32/', 'final/')

df['RoBERTa-GT'] = prepare_plm_file(path+'roberta/ground/')
df['RoBERTa-GT-UMC'] = prepare_plm_file(path2+'umc/roberta/ground/', 'final/')
df['RoBERTa-GT-SEL'] = prepare_plm_file(path2+'select/roberta/ground/', 'final/')
df['RoBERTa-LLM'] = prepare_plm_file(path+'roberta/qwen_32/')
df['RoBERTa-LLM-UMC'] = prepare_plm_file(path2+'umc/roberta/qwen_32/', 'final/')
df['RoBERTa-LLM-SEL'] = prepare_plm_file(path2+'select/roberta/qwen_32/', 'final/')
df['RoBERTa-SLM'] = prepare_plm_file(path+'roberta/roberta/qwen_32/')
df['RoBERTa-SLM-UMC'] = prepare_plm_file(path2+'umc/roberta/roberta/qwen_32/', 'final/')
df['RoBERTa-SLM-SEL'] = prepare_plm_file(path2+'select/roberta/roberta/qwen_32/', 'final/')

df = prepare_df(df)

latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)