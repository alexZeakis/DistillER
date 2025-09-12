import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_plm_file, prepare_umc_file, prepare_df

################### COMPARING PLMS - GENERALIZED ON TEST DATA #################

path = '../../../log/matching/slm/'
df = pd.DataFrame()
df['UMC'] = prepare_umc_file('../../../log/matching/baselines/umc/')

df['SMiniLM-GT'] = prepare_plm_file(path+'sminilm/ground/')
df['SMiniLM-Qwen2.5:32b'] = prepare_plm_file(path+'sminilm/qwen_32/')
df['SMiniLM-Hybrid'] = prepare_plm_file(path+'sminilm/roberta/qwen_32/')

df['RoBERTa-GT'] = prepare_plm_file(path+'roberta/ground/')
df['RoBERTa-Qwen2.5:32b'] = prepare_plm_file(path+'roberta/qwen_32/')
df['RoBERTa-Hybrid'] = prepare_plm_file(path+'roberta/roberta/qwen_32/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)