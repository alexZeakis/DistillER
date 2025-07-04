import pandas as pd
from eval_utils import prepare_plm_file, prepare_ft_file, prepare_umc_file, prepare_df
from time import time
import os

################### COMPARING PLMS - GENERALIZED ON TEST DATA #################

t1 = time()
path = '../../log/matching/hybrid/'
df = pd.DataFrame()
df['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')

slm_annotators={
                # 'GT': 'ground',
                'LLM': 'qwen_32',
                # 'SLM': 'roberta/qwen_32'
                }
                
slm_models={
    'SMiniLm': 'sminilm',
    'RoBERTa': 'roberta'}

llm_models={'FT': 'llm/qwen_32',
            'PT': 'baselines/pretrained'}

for slm_mod_k, slm_mod_val in slm_models.items():
    for slm_an_k, slm_an_val in slm_annotators.items():
        for llm_mod_k, llm_mod_val in llm_models.items():
            t_path = os.path.join(path, slm_mod_val, slm_an_val, llm_mod_val)+"/"
            print(t_path)
            t_name = '{}-{}/{}'.format(slm_mod_k, slm_an_k, llm_mod_k)
            # t_name = '{}/{}'.format(slm_an_k, llm_mod_k)
            df[t_name] = prepare_plm_file(t_path)

path = '../../log/matching/disambiguation/'
df['SMiniLM-LLM'] = prepare_plm_file(path+'select/sminilm/qwen_32/', 'final/')
df['RoBERTa-LLM'] = prepare_plm_file(path+'select/roberta/qwen_32/', 'final/')

df = prepare_df(df)
cols = ['UMC', 'SMiniLM-LLM', 'SMiniLm-LLM/FT', 'SMiniLm-LLM/PT', 
        'RoBERTa-LLM', 'RoBERTa-LLM/FT', 'RoBERTa-LLM/PT']
df = df[cols]

latex_code = df.to_latex(index=True, escape=False, multirow=False)

t2 = time()
print('Time elapsed {:.2f}'.format(t2-t1))