import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import get_llm_testing_time, get_llm_training_time, \
    get_plm_time, get_plm_student_time, get_inference_time

################### COMPARING LLMS ON TEST DATA #########################

total_df = pd.DataFrame()

path = '../../../log/matching/sft/'
bpath = '../../../log/matching/baselines/pretrained/'
annot_path = '../../../log/matching/annotate/'

# ################### COMPARING PLMS - GENERALIZED ON TEST DATA #################
df = {}

llm_teacher = get_inference_time(annot_path+'llm/qwen_32/partial_responses/')
slm_teacher = llm_teacher * 0.2 + get_plm_time(annot_path+'slm/llm/roberta/qwen_32/')

df['Llama-PT'] = get_inference_time(bpath + 'llama3.1:8b/', tosum=False)
df['Teacher'] = get_inference_time(bpath + 'qwen2.5:32b/', tosum=False)

df['Llama-on-LLM'] = get_llm_testing_time(path+'llm/qwen_32/', tosum=False)
df['Llama-on-LLM']['train'] = get_llm_training_time(path+'llm/qwen_32/')
df['Llama-on-LLM']['teacher'] = llm_teacher

df['Llama-on-SLM'] = get_llm_testing_time(path+'llm/roberta/qwen_32/', tosum=False)
df['Llama-on-SLM']['train'] = get_llm_training_time(path+'llm/roberta/qwen_32/')
df['Llama-on-SLM']['teacher'] = slm_teacher

df['RoBERTa-on-LLM'] = get_plm_student_time(path+'slm/roberta/qwen_32/')
df['RoBERTa-on-LLM']['teacher'] = llm_teacher

df['RoBERTa-on-SLM'] = get_plm_student_time(path+'slm/roberta/roberta/qwen_32/')
df['RoBERTa-on-SLM']['teacher'] = slm_teacher

df['SMiniLM-on-LLM'] = get_plm_student_time(path+'slm/sminilm/qwen_32/')
df['SMiniLM-on-LLM']['teacher'] = llm_teacher

df['SMiniLM-on-SLM'] = get_plm_student_time(path+'slm/sminilm/roberta/qwen_32/')
df['SMiniLM-on-SLM']['teacher'] = slm_teacher
df = pd.DataFrame(df)

df = df.loc[['teacher', 'train', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']]
# df = df.round(2)
df.loc['Sum', :] = df.sum()
df = df.fillna(0).astype(int)

latex_code = df.to_latex(index=True, escape=False, multirow=False)