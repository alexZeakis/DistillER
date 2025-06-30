import pandas as pd
from eval_utils import prepare_plm_train_file, prepare_df

path = '../../log/matching/annotate_hybrid/llm/'
df = pd.DataFrame()
df['SMiniLM-8b'] = prepare_plm_train_file(path+'sminilm/llama_8/')
df['SMiniLM-70b'] = prepare_plm_train_file(path+'sminilm/llama_70/')
df['SMiniLM-14'] = prepare_plm_train_file(path+'sminilm/qwen_14/')
df['SMiniLM-32'] = prepare_plm_train_file(path+'sminilm/qwen_32/')
df['RoBERTa-8b'] = prepare_plm_train_file(path+'roberta/llama_8/')
df['RoBERTa-70b'] = prepare_plm_train_file(path+'roberta/llama_70/')
df['RoBERTa-14'] = prepare_plm_train_file(path+'roberta/qwen_14/')
df['RoBERTa-32'] = prepare_plm_train_file(path+'roberta/qwen_32/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)


path = '../../log/matching/annotate_hybrid/size/'
df = pd.DataFrame()
df['SMiniLM-5%'] = prepare_plm_train_file(path+'sminilm/0.95/')
df['SMiniLM-10%'] = prepare_plm_train_file(path+'sminilm/0.90/')
df['SMiniLM-20%'] = prepare_plm_train_file(path+'sminilm/0.80/')
df['RoBERTa-5%'] = prepare_plm_train_file(path+'roberta/0.95/')
df['RoBERTa-10%'] = prepare_plm_train_file(path+'roberta/0.90/')
df['RoBERTa-20%'] = prepare_plm_train_file(path+'roberta/0.80/')

df = prepare_df(df)
latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)

path = '../../log/matching/annotate_hybrid/llm/'
path2 = '../../log/matching/annotate_hybrid/ground/'
df = pd.DataFrame()
df['SMiniLM-Ground'] = prepare_plm_train_file(path2+'sminilm/')
df['SMiniLM-32'] = prepare_plm_train_file(path+'sminilm/qwen_32/')
df['RoBERTa-Ground'] = prepare_plm_train_file(path2+'roberta/')
df['RoBERTa-32'] = prepare_plm_train_file(path+'roberta/qwen_32/')

df = prepare_df(df)
latex_code_3 = df.to_latex(index=True, escape=False, multirow=False)