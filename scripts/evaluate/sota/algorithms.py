import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_ft_file, prepare_df, prepare_pt_file, prepare_comem_file
import json

def read_sota(path, case, metric='f1'):
    scores = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            if metric not in j:
                continue
            scores[j[case]] = j[metric]
    return pd.Series(scores)

def read_sudowoodo_sota(path):
    scores = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            data = j['run_tag'].split("_")[1]
            scores[data] = j['t_f1'] # will keep f1 of last epoch
    return pd.Series(scores)

################### COMPARING PLMS - GENERALIZED ON TEST DATA #################

path = '../../../log/matching/sft/llm/'
df = pd.DataFrame()

df['DistillER-SFT/LLM'] = prepare_ft_file(path+'qwen_32/test_responses/')
df['DistillER-SFT/SLM'] = prepare_ft_file(path+'roberta/qwen_32/test_responses/')

df['ComEM'] = prepare_comem_file('../../../log/matching/sota/ComEM/responses/SELECT/',
                                 gt_dir='../../../log/matching/baselines/pretrained/llama3.1:8b/')

df['ZeroER'] = read_sota('../../../log/matching/sota/zeroer/scores.jsonl', 'dataset')
df['CollaborEM'] = read_sota('../../../log/matching/sota/CollaborEM.txt', 'case', 'test_f1')

df['SudoWoodo'] = read_sudowoodo_sota('../../../log/matching/sota/sudowoodo/sudowoodo.txt')
df['HierGAT'] = read_sota('../../../log/matching/sota/HierGAT.jsonl', 'data_name')
df['Unicorn'] = read_sota('../../../log/matching/sota/Unicorn.jsonl', 'dataset')

df = prepare_df(df)

latex_code = df.to_latex(index=True, escape=False, multirow=False)