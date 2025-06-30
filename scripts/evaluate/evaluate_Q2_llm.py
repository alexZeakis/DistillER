import pandas as pd
from eval_utils import prepare_pt_file,  prepare_df

################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../log/matching/annotate/blocking/'
df = pd.DataFrame()
df['Llama3.1:8b'] = prepare_pt_file(path+'llama_8/partial_responses/')
df['Llama3.1:70b'] = prepare_pt_file(path+'llama_70/partial_responses/')
# df['Qwen2.5:7b'] = prepare_pt_file(path+'qwen_7/partial_responses/')
df['Qwen2.5:14b'] = prepare_pt_file(path+'qwen_14/partial_responses/')
df['Qwen2.5:32b'] = prepare_pt_file(path+'qwen_32/partial_responses/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)


