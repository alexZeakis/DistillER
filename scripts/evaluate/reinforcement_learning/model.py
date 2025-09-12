import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_umc_file, prepare_pt_file, prepare_ft_file, prepare_df


################### COMPARING LLMS ON TEST DATA #########################

df = pd.DataFrame()

path = '../../../log/matching/rl/'
df['UMC'] = prepare_umc_file('../../../log/matching/baselines/umc/')
df['Pretrained'] = prepare_pt_file('../../../log/matching/baselines/pretrained/')
df['SFT'] = prepare_ft_file('../../../log/matching/llm/qwen_32/test_responses/')

df['LLM-GRPO-PT'] = prepare_ft_file(path+'grpo/PT/qwen_32/test_responses/')
df['LLM-GRPO-SFT'] = prepare_ft_file(path+'grpo/SFT/qwen_32/test_responses/')

df['LLM-DPO-PT'] = prepare_ft_file(path+'dpo/PT/qwen_32/test_responses/')
df['LLM-DPO-SFT'] = prepare_ft_file(path+'dpo/SFT/qwen_32/test_responses/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)