import pandas as pd
from eval_utils import prepare_plm_file, prepare_df
import json

comem_comem = {'D2': 87.62, 'D3': 69.63, 'D4': 90.85, 'D5': 96.74,
               'D6': 84.16, 'D7': 84.82, 'D8': 86.37, 'D9': 84.68,}
avenger_unsup = {'D2': 0.93, 'D3': 0.66, 'D4': 0.91, 'D5': 0.77,
                 'D6': 0.67, 'D7': 0.77, 'D8': 0.82, 'D9': 0.85}
avenger_sup = {'D2': 0.94, 'D3': 0.68, 'D4': 0.99, 'D5': 0.90,
               'D6': 0.79, 'D7': 0.90, 'D8': 0.89, 'D9': 0.96,}



################### COMPARING PLMS - GENERALIZED ON TEST DATA #################

path2 = '../../log/matching/disambiguation/'
df = pd.DataFrame()

df['RoBERTa-GT-UMC'] = prepare_plm_file(path2+'umc/roberta/ground/', 'final/')
df['ComEM'] = pd.Series(comem_comem) / 100
df['Avenger-Unsup.'] = pd.Series(avenger_unsup)
df['Avenger-Sup.'] = pd.Series(avenger_sup)

zeroer_scores = {}
with open('../../log/matching/sota/zeroer/scores.jsonl') as f:
    for line in f:
        j = json.loads(line)
        zeroer_scores[j['dataset']] = j['f1']
df['ZeroER'] = pd.Series(zeroer_scores)

df = prepare_df(df)

latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)