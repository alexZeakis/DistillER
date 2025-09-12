import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_umc_file, prepare_pt_file, prepare_ft_file, \
    prepare_df, prepare_ft_file_time, prepare_df_time


################### COMPARING LLMS ON TEST DATA #########################

df = pd.DataFrame()

path = '../../../log/matching/rl/'
df['UMC'] = prepare_umc_file('../../../log/matching/baselines/umc/')
df['Pretrained'] = prepare_pt_file('../../../log/matching/baselines/pretrained/')    

df['Ground-GRPO'] = prepare_ft_file(path+'grpo/PT/ground/test_responses/')
df['LLM-GRPO'] = prepare_ft_file(path+'grpo/PT/qwen_32/test_responses/')
df['SLM-GRPO'] = prepare_ft_file(path+'grpo/PT/roberta/qwen_32/test_responses/')

df['Ground-DPO'] = prepare_ft_file(path+'dpo/PT/ground/test_responses/')
df['LLM-DPO'] = prepare_ft_file(path+'dpo/PT/qwen_32/test_responses/')
df['SLM-DPO'] = prepare_ft_file(path+'dpo/PT/roberta/qwen_32/test_responses/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)


df = pd.DataFrame()

path = '../../../log/matching/rl/'

df['Ground-GRPO'] = prepare_ft_file_time(path+'grpo/PT/ground/test_responses/')
df['LLM-GRPO'] = prepare_ft_file_time(path+'grpo/PT/qwen_32/test_responses/')
df['SLM-GRPO'] = prepare_ft_file_time(path+'grpo/PT/roberta/qwen_32/test_responses/')

df['Ground-DPO'] = prepare_ft_file_time(path+'dpo/PT/ground/test_responses/')
df['LLM-DPO'] = prepare_ft_file_time(path+'dpo/PT/qwen_32/test_responses/')
df['SLM-DPO'] = prepare_ft_file_time(path+'dpo/PT/roberta/qwen_32/test_responses/')

df = prepare_df_time(df)
latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)