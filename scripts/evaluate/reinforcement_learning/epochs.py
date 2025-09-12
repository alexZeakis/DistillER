import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_umc_file, prepare_pt_file, prepare_ft_file, prepare_df


################### COMPARING LLMS ON TEST DATA #########################

df = pd.DataFrame()

path = '../../../log/matching/rl/'
df['UMC'] = prepare_umc_file('../../../log/matching/baselines/umc/')
df['Pretrained'] = prepare_pt_file('../../../log/matching/baselines/pretrained/')    

df['LLM-GRPO-1'] = prepare_ft_file(path+'grpo/qwen_32/test_responses/')
df['LLM-GRPO-3'] = prepare_ft_file(path+'grpo/epochs/qwen_32/test_responses/')

df['LLM-DPO-1'] = prepare_ft_file(path+'dpo/qwen_32/test_responses/')
df['LLM-DPO-3'] = prepare_ft_file(path+'dpo/epochs/qwen_32/test_responses/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)