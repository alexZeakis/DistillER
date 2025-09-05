import pandas as pd
from eval_utils import prepare_ft_file, prepare_df
import json

comem_comem = {'D2': 87.62, 'D3': 69.63, 'D4': 90.85, 'D5': 96.74,
               'D6': 84.16, 'D7': 84.82, 'D8': 86.37, 'D9': 84.68,}
avenger_unsup = {'D2': 0.93, 'D3': 0.66, 'D4': 0.91, 'D5': 0.77,
                 'D6': 0.67, 'D7': 0.77, 'D8': 0.82, 'D9': 0.85}
avenger_sup = {'D2': 0.94, 'D3': 0.68, 'D4': 0.99, 'D5': 0.90,
               'D6': 0.79, 'D7': 0.90, 'D8': 0.89, 'D9': 0.96,}

def read_sota(path, case, metric='f1'):
    scores = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
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

# path2 = '../../log/matching/disambiguation/'
path = '../../log/matching/llm/'
df = pd.DataFrame()

# df['RoBERTa-GT-UMC'] = prepare_plm_file(path2+'umc/roberta/ground/', 'final/')
df['DistillER-LLM Labels'] = prepare_ft_file(path+'qwen_32/test_responses/')
df['DistillER-SLM Labels'] = prepare_ft_file(path+'roberta/qwen_32/test_responses/')
# df['ComEM'] = pd.Series(comem_comem) / 100
# df['Avenger-Unsup.'] = pd.Series(avenger_unsup)
# df['Avenger-Sup.'] = pd.Series(avenger_sup)
df['AvengER'] = prepare_ft_file(path+'ground/test_responses/')

df['ZeroER'] = read_sota('../../log/matching/sota/zeroer/scores.jsonl', 'dataset')
df['CollaborEM'] = read_sota('../../log/matching/sota/CollaborEM.txt', 'case', 'test_f1')
df['HierGAT'] = read_sota('../../log/matching/sota/HierGAT.jsonl', 'data_name')
df['Unicorn'] = read_sota('../../log/matching/sota/Unicorn.jsonl', 'data')
df['SudoWoodo'] = read_sudowoodo_sota('../../log/matching/sota/sudowoodo/sudowoodo.txt')



df = prepare_df(df)

latex_code = df.to_latex(index=True, escape=False, multirow=False)