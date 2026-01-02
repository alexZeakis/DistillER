import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import os
from eval_utils import prepare_pt_file, prepare_df

def prepare_distr(path, label):
    scores = {}
    for file in os.listdir(path):
        if not file.endswith('csv'):
            continue
        df = pd.read_csv(path+file)
        df['Label'] = df.D2 == df['True']
        df2 = df.groupby('D1')['Label'].sum()
        
        if label == 'positive':
            scores[file[:2]] = (df2 > 0).sum()
        else:
            scores[file[:2]] = (df2 == 0).sum()
    return pd.Series(scores)



################### COMPARING LLMS ON TRAIN DATA AND METHOD OF SAMPLING #########################

methods = ['random', 'blocking_max', 'blocking_top2',
           'clustering_hierarchical', 'clustering_kmeans', 
           'sampled']


path = '../../../log/matching/data_selection/'
df = pd.DataFrame()
for method in methods:
    for model in ['llama_8', 'qwen_14']:
        df[f'{method}-{model}'] = prepare_pt_file(path+f'{method}/{model}/partial_responses/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)

################### DISTRIBUTION OF CLASSES PER METHOD OF SAMPLING #########################

path = '../../../data/ccer/cleaned/'

scores = pd.DataFrame()
for method in methods:
    for label in ['positive', 'negative']:
        scores[f'{label}-{method}'] = prepare_distr(path+f'fine_tuning/{method}/train/', label)

scores = prepare_df(scores)
latex_code_2 = scores.to_latex(index=True, escape=False, multirow=False)


scores_2 = pd.DataFrame()
for method in methods:
    scores_2[method] = scores[f'positive-{method}'] / (scores[f'positive-{method}'] + scores[f'negative-{method}'])    

scores_2 = prepare_df(scores_2)
latex_code_3 = scores_2.to_latex(index=True, escape=False, multirow=False)