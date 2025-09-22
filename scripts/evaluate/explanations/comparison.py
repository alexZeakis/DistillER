import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_umc_file, prepare_pt_file, prepare_ft_file, prepare_df


################### COMPARING LLMS ON TEST DATA #########################

df = pd.DataFrame()

path = '../../../log/matching/'
df['UMC'] = prepare_umc_file('../../../log/matching/baselines/umc/')
df['Pretrained'] = prepare_pt_file('../../../log/matching/baselines/pretrained/')    

df['SFT-Answer'] = prepare_ft_file(path+'llm/qwen_32/test_responses/')
df['SFT-Expl.'] = prepare_ft_file(path+'justifications/llm/qwen_32/test_responses/')

# df['GRPO-Answer'] = prepare_ft_file(path+'rl/grpo/PT/qwen_32/test_responses/')
# df['GRPO-PT-Expl.'] = 0
# df['GRPO-FT-Expl.'] = 0
# df['GRPO-PT-Expl.'] = prepare_ft_file(path+'justifications/grpo/PT/qwen_32/test_responses/')
# df['GRPO-FT-Expl.'] = prepare_ft_file(path+'justifications/grpo/FT/qwen_32/test_responses/')

df['DPO-Answer'] = prepare_ft_file(path+'rl/dpo/PT/qwen_32/test_responses/')
df['DPO-PT-Expl.'] = prepare_ft_file(path+'justifications/dpo/PT/qwen_32/test_responses/')
df['DPO-FT-Expl.'] = prepare_ft_file(path+'justifications/dpo/FT/qwen_32/test_responses/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)
