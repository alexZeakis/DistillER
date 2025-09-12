import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_train_file, prepare_df

path = '../../../log/matching/annotate_hybrid/llm/'
gt_path = '../../../log/matching/annotate/blocking/ground/partial/'
df = pd.DataFrame()
df['SMiniLM-8b'] = prepare_train_file(path+'sminilm/llama_8/', gt_path)
df['SMiniLM-70b'] = prepare_train_file(path+'sminilm/llama_70/', gt_path)
df['SMiniLM-14'] = prepare_train_file(path+'sminilm/qwen_14/', gt_path)
df['SMiniLM-32'] = prepare_train_file(path+'sminilm/qwen_32/', gt_path)
df['RoBERTa-8b'] = prepare_train_file(path+'roberta/llama_8/', gt_path)
df['RoBERTa-70b'] = prepare_train_file(path+'roberta/llama_70/', gt_path)
df['RoBERTa-14'] = prepare_train_file(path+'roberta/qwen_14/', gt_path)
df['RoBERTa-32'] = prepare_train_file(path+'roberta/qwen_32/', gt_path)

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)


path = '../../../log/matching/annotate_hybrid/size/'
df = pd.DataFrame()
df['SMiniLM-5%'] = prepare_train_file(path+'sminilm/0.95/', gt_path)
df['SMiniLM-10%'] = prepare_train_file(path+'sminilm/0.90/', gt_path)
df['SMiniLM-20%'] = prepare_train_file(path+'sminilm/0.80/', gt_path)
df['RoBERTa-5%'] = prepare_train_file(path+'roberta/0.95/', gt_path)
df['RoBERTa-10%'] = prepare_train_file(path+'roberta/0.90/', gt_path)
df['RoBERTa-20%'] = prepare_train_file(path+'roberta/0.80/', gt_path)

df = prepare_df(df)
latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)

path = '../../../log/matching/annotate_hybrid/llm/'
path2 = '../../../log/matching/annotate_hybrid/ground/'
df = pd.DataFrame()
df['SMiniLM-Ground'] = prepare_train_file(path2+'sminilm/', gt_path)
df['SMiniLM-32'] = prepare_train_file(path+'sminilm/qwen_32/', gt_path)
df['RoBERTa-Ground'] = prepare_train_file(path2+'roberta/', gt_path)
df['RoBERTa-32'] = prepare_train_file(path+'roberta/qwen_32/', gt_path)

df = prepare_df(df)
latex_code_3 = df.to_latex(index=True, escape=False, multirow=False)