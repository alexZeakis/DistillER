import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import get_llm_testing_time, get_llm_training_time, \
    get_plm_time, get_plm_student_time, get_inference_time
import json

def read_sota(path, case, metrics):
    scores = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            if case not in j:
                continue
            scores[j[case]] = 0
            for metric in metrics:
                scores[j[case]] += j[metric]
    return scores

def read_single_sota(path, metric):
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            if metric in j:
                return j[metric]

################### COMPARING LLMS ON TEST DATA #########################

total_df = pd.DataFrame()

path = '../../../log/matching/sft/'
bpath = '../../../log/matching/baselines/pretrained/'
annot_path = '../../../log/matching/annotate/'

# ################### COMPARING PLMS - GENERALIZED ON TEST DATA #################
df = {}

llm_teacher = get_inference_time(annot_path+'llm/qwen_32/partial_responses/')
slm_teacher = llm_teacher * 0.2 + get_plm_time(annot_path+'slm/llm/roberta/qwen_32/')

df['Llama-on-LLM'] = get_llm_testing_time(path+'llm/qwen_32/', tosum=False)
df['Llama-on-LLM']['train'] = get_llm_training_time(path+'llm/qwen_32/')
df['Llama-on-LLM']['teacher'] = llm_teacher

df['Llama-on-SLM'] = get_llm_testing_time(path+'llm/roberta/qwen_32/', tosum=False)
df['Llama-on-SLM']['train'] = get_llm_training_time(path+'llm/roberta/qwen_32/')
df['Llama-on-SLM']['teacher'] = slm_teacher

comem_sel = get_inference_time('../../../log/matching/sota/ComEM/responses/SELECT/',
                               tosum=False, checkfile=False)
comem_mat = get_inference_time('../../../log/matching/sota/ComEM/responses/MATCH/',
                               tosum=False, checkfile=False)
df['ComEM'] = {k: comem_sel[k] + comem_mat[k] for k in comem_sel.keys() }


df['ZeroER'] = read_sota('../../../log/matching/sota/zeroer/scores.jsonl', 
                         'dataset', ['features_time', 'zeroer_time'])

df['CollaborEM'] = read_sota('../../../log/matching/sota/CollaborEM.txt', 
                             'case', ['test_time', 'train_time'])

df['SudoWoodo'] = {}

df['HierGAT'] = read_sota('../../../log/matching/sota/HierGAT.jsonl',
                          'data_name', ['testing_time'])
df['HierGAT']['train'] = sum(read_sota('../../../log/matching/sota/HierGAT.jsonl',
                          'data_name', ['training_time']).values())
df['HierGAT']['teacher'] = llm_teacher

df['Unicorn'] = read_sota('../../../log/matching/sota/Unicorn.jsonl',
                          'dataset', ['time'])
df['Unicorn']['teacher'] = llm_teacher
df['Unicorn']['train'] = read_single_sota('../../../log/matching/sota/Unicorn.jsonl', 'training_time')


df = pd.DataFrame(df)

df = df.loc[['teacher', 'train', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']]
# df = df.round(2)
df.loc['Sum', :] = df.sum()
df = df.fillna(0).astype(int)

latex_code = df.to_latex(index=True, escape=False, multirow=False)