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

path = '../../log/matching/annotate/'
df = pd.DataFrame()
df['Random-Llama3.1:8b'] = prepare_pt_file(path+'random/llama_8/partial_responses/')
df['Random-Qwen2.5:14b'] = prepare_pt_file(path+'random/qwen_14/partial_responses/')

df['Blocking-Max-Llama3.1:8b'] = prepare_pt_file(path+'blocking/llama_8/partial_responses/')
df['Blocking-Max-Qwen2.5:14b'] = prepare_pt_file(path+'blocking/qwen_14/partial_responses/')

df['Blocking-Top2-Llama3.1:8b'] = prepare_pt_file(path+'blocking_top2/llama_8/partial_responses/')
df['Blocking-Top2-Qwen2.5:14b'] = prepare_pt_file(path+'blocking_top2/qwen_14/partial_responses/')

df['Clustering-Hier-Llama3.1:8b'] = prepare_pt_file(path+'clustering_hierarchical/llama_8/partial_responses/')
df['Clustering-Hier-Qwen2.5:14b'] = prepare_pt_file(path+'clustering_hierarchical/qwen_14/partial_responses/')

df['Clustering-KMeans-Llama3.1:8b'] = prepare_pt_file(path+'clustering_kmeans/llama_8/partial_responses/')
df['Clustering-KMeans-Qwen2.5:14b'] = prepare_pt_file(path+'clustering_kmeans/qwen_14/partial_responses/')

df['Sampled-Llama3.1:8b'] = prepare_pt_file(path+'sampled/llama_8/partial_responses/')
df['Sampled-Qwen2.5:14b'] = prepare_pt_file(path+'sampled/qwen_14/partial_responses/')

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)


################### DISTRIBUTION OF CLASSES PER METHOD OF SAMPLING #########################

path = '../../data/ccer/cleaned/'

scores = pd.DataFrame()
scores['Positive-Random'] = prepare_distr(path+'fine_tuning/random/train/', 'positive')
scores['Negative-Random'] = prepare_distr(path+'fine_tuning/random/train/', 'negative')
scores['Positive-Blocking-Max'] = prepare_distr(path+'fine_tuning/blocking_max/train/', 'positive')
scores['Negative-Blocking-Max'] = prepare_distr(path+'fine_tuning/blocking_max/train/', 'negative')
scores['Positive-Blocking-Top2'] = prepare_distr(path+'fine_tuning/blocking_top2/train/', 'positive')
scores['Negative-Blocking-Top2'] = prepare_distr(path+'fine_tuning/blocking_top2/train/', 'negative')
scores['Positive-Clustering-Hier'] = prepare_distr(path+'fine_tuning/clustering_hierarchical/train/', 'positive')
scores['Negative-Clustering-Hier'] = prepare_distr(path+'fine_tuning/clustering_hierarchical/train/', 'negative')
scores['Positive-Clustering-KMeans'] = prepare_distr(path+'fine_tuning/clustering_kmeans/train/', 'positive')
scores['Negative-Clustering-KMeans'] = prepare_distr(path+'fine_tuning/clustering_kmeans/train/', 'negative')
scores['Positive-Sampled'] = prepare_distr(path+'fine_tuning/sampled/train/', 'positive')
scores['Negative-Sampled'] = prepare_distr(path+'fine_tuning/sampled/train/', 'negative')

scores = prepare_df(scores)
latex_code_2 = scores.to_latex(index=True, escape=False, multirow=False)