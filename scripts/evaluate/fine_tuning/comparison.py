import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_plm_file, prepare_ft_file, prepare_umc_file, prepare_df

################### COMPARING PLMS - GENERALIZED ON TEST DATA #################

path = '../../../log/matching/slm/'
df = pd.DataFrame()
df['UMC'] = prepare_umc_file('../../../log/matching/baselines/umc/')

path = '../../../log/matching/llm/'
df['FT LLM-LLM Labels'] = prepare_ft_file(path+'qwen_32/test_responses/')
df['FT LLM-SLM Labels'] = prepare_ft_file(path+'roberta/qwen_32/test_responses/')

path = '../../../log/matching/disambiguation/'
df['FT RoBERTa-LLM Labels (SEL)'] = prepare_plm_file(path+'select/roberta/qwen_32/', 'final/')

# path = '../../../log/matching/hybrid/roberta/qwen_32/baselines/pretrained/'
# df['FT Hybrid - LLM Labels'] = prepare_plm_file(path)

df = prepare_df(df)

latex_code = df.to_latex(index=True, escape=False, multirow=False)

