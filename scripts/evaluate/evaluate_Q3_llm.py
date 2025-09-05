import pandas as pd
from eval_utils import prepare_umc_file, prepare_pt_file, prepare_ft_file, prepare_df


################### COMPARING LLMS ON TEST DATA #########################

df = pd.DataFrame()

path = '../../log/matching/llm/'
df['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
df['Pretrained'] = prepare_pt_file('../../log/matching/baselines/pretrained/')    
df['GT'] = prepare_ft_file(path+'ground/test_responses/')
df['Qwen2.5:32b'] = prepare_ft_file(path+'qwen_32/test_responses/')
df['RoBERTa'] = prepare_ft_file(path+'roberta/qwen_32/test_responses/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)