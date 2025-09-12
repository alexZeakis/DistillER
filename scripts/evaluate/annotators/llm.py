import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_train_file,  prepare_df

################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../../log/matching/annotate/blocking/'
gt_path = '../../../log/matching/annotate/blocking/ground/partial/'
df = pd.DataFrame()
df['Llama3.1:8b'] = prepare_train_file(path+'llama_8/', gt_path)
df['Llama3.1:70b'] = prepare_train_file(path+'llama_70/', gt_path)
df['Qwen2.5:14b'] = prepare_train_file(path+'qwen_14/', gt_path)
df['Qwen2.5:32b'] = prepare_train_file(path+'qwen_32/', gt_path)
df['Multi-Teacher'] = prepare_train_file(path+'multi_llm/', gt_path)

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)