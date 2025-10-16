import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import os
from eval_utils import prepare_pt_file, prepare_df
import matplotlib.pyplot as plt

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



def plot_model_boxplots(df, output_prefix='boxplot', font_size=14):
    """
    Create boxplots for Llama3.1 and Qwen2.5 columns in a DataFrame, save as PDF.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        output_prefix (str): Prefix for saved PDF files.
        font_size (int): Font size for labels and ticks.
    """
    # Remove 'Mean' index
    df = df.drop('Mean', axis=0)
    
    # Separate columns by model
    llama_cols = [col for col in df.columns if 'Llama3.1:8b' in col]
    qwen_cols = [col for col in df.columns if 'Qwen2.5:14b' in col]

    df_llama = df[llama_cols].copy()
    df_qwen = df[qwen_cols].copy()
    
    # Remove model suffix
    df_llama.columns = [col.replace('-Llama3.1:8b', '') for col in df_llama.columns]
    df_qwen.columns = [col.replace('-Qwen2.5:14b', '') for col in df_qwen.columns]

    # Update font size globally
    plt.rcParams.update({'font.size': font_size})
    
    # Plot Llama
    plt.figure(figsize=(10,6))
    bp_llama = df_llama.boxplot(grid=False)  # <-- remove gridlines
    # df_llama.boxplot()
    plt.xticks(rotation=30)
    # plt.title('Llama3.1:8b Results')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_llama.pdf')
    plt.show()
    plt.close()
    
    # Plot Qwen
    plt.figure(figsize=(10,6))
    bp_qwen = df_qwen.boxplot(grid=False)  # <-- remove gridlines
    # df_qwen.boxplot()
    plt.xticks(rotation=30)
    # plt.title('Qwen2.5:14b Results')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_qwen.pdf')
    plt.show()
    plt.close()
    
    
    print(f"Boxplots saved as {output_prefix}_llama.pdf and {output_prefix}_qwen.pdf")

################### COMPARING LLMS ON TRAIN DATA AND METHOD OF SAMPLING #########################

path = '../../../log/matching/data_selection/'
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

plot_model_boxplots(df)

################### DISTRIBUTION OF CLASSES PER METHOD OF SAMPLING #########################

path = '../../../data/ccer/cleaned/'

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