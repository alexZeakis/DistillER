import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval_utils import prepare_plm_file_cardinality, prepare_plm_file, prepare_ft_file, prepare_umc_file, prepare_df
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ordered_bars(values, bar_names, segment_labels=None, colors=None, title='Bar Plot'):
    """
    Plots bars with segments overlaid in ascending order for each bar, all starting from zero.

    Args:
        values: list of 3 lists, each of length N (3xN array of values for each segment)
        bar_names: list of N names for x-axis
        segment_labels: list of 3 labels for the segments (for legend)
        colors: list of 3 colors for the segments
        title: plot title
    """
    
    plt.rcParams.update({'font.size': 14}) 
    values = np.array(values)  # shape: (3, N)
    N = len(bar_names)
    if values.shape[1] != N:
        raise ValueError("All segments must have the same length as bar_names")
    
    if colors is None:
        colors = ['#E41A1C', '#377EB8', '#4DAF4A']  # Red, Blue, Green
    if segment_labels is None:
        segment_labels = ['Segment 1', 'Segment 2', 'Segment 3']
    
    fig, ax = plt.subplots(figsize=(8,5))
    
    for i in range(N):  # iterate over each bar
        vals = values[:, i]
        order = np.argsort(vals)[::-1]  # largest first
        for idx in order:
            ax.bar(bar_names[i], vals[idx], color=colors[idx], alpha=0.8)
    
    # Add legend
    for c, label in zip(colors, segment_labels):
        ax.bar(0, 0, color=c, alpha=0.8, label=label)  # dummy bars for legend
    ax.legend()
    
    ax.set_ylim(0.50, 0.75)
    ax.set_ylabel('F1-score')
    ax.set_xticklabels(bar_names, rotation=45)
    ax.set_title(title)
    plt.savefig('disambiguation.pdf', bbox_inches='tight')
    plt.show()

# ################### COMPARING PLMS - GENERALIZED ON TEST DATA #################

path = '../../../log/matching/slm/'
df = pd.DataFrame()

df['SMiniLM-GT'] = prepare_plm_file_cardinality(path+'sminilm/ground/')
df['SMiniLM-LLM'] = prepare_plm_file_cardinality(path+'sminilm/qwen_32/')
df['SMiniLM-SLM'] = prepare_plm_file_cardinality(path+'sminilm/roberta/qwen_32/')
df['RoBERTa-GT'] = prepare_plm_file_cardinality(path+'roberta/ground/')
df['RoBERTa-LLM'] = prepare_plm_file_cardinality(path+'roberta/qwen_32/')
df['RoBERTa-SLM'] = prepare_plm_file_cardinality(path+'roberta/roberta/qwen_32/')

# df = prepare_df(df)
df = df.T

latex_code = df.to_latex(index=True, escape=False, multirow=False)


path = '../../../log/matching/slm/'
path2 = '../../../log/matching/disambiguation/'
df = pd.DataFrame()

df['SMiniLM-GT'] = prepare_plm_file(path+'sminilm/ground/')
df['SMiniLM-GT-UMC'] = prepare_plm_file(path2+'umc/sminilm/ground/', 'final/')
df['SMiniLM-GT-SEL'] = prepare_plm_file(path2+'select/sminilm/ground/', 'final/')
df['SMiniLM-LLM'] = prepare_plm_file(path+'sminilm/qwen_32/')
df['SMiniLM-LLM-UMC'] = prepare_plm_file(path2+'umc/sminilm/qwen_32/', 'final/')
df['SMiniLM-LLM-SEL'] = prepare_plm_file(path2+'select/sminilm/qwen_32/', 'final/')
df['SMiniLM-SLM'] = prepare_plm_file(path+'sminilm/roberta/qwen_32/')
df['SMiniLM-SLM-UMC'] = prepare_plm_file(path2+'umc/sminilm/roberta/qwen_32/', 'final/')
df['SMiniLM-SLM-SEL'] = prepare_plm_file(path2+'select/sminilm/roberta/qwen_32/', 'final/')

df['RoBERTa-GT'] = prepare_plm_file(path+'roberta/ground/')
df['RoBERTa-GT-UMC'] = prepare_plm_file(path2+'umc/roberta/ground/', 'final/')
df['RoBERTa-GT-SEL'] = prepare_plm_file(path2+'select/roberta/ground/', 'final/')
df['RoBERTa-LLM'] = prepare_plm_file(path+'roberta/qwen_32/')
df['RoBERTa-LLM-UMC'] = prepare_plm_file(path2+'umc/roberta/qwen_32/', 'final/')
df['RoBERTa-LLM-SEL'] = prepare_plm_file(path2+'select/roberta/qwen_32/', 'final/')
df['RoBERTa-SLM'] = prepare_plm_file(path+'roberta/roberta/qwen_32/')
df['RoBERTa-SLM-UMC'] = prepare_plm_file(path2+'umc/roberta/roberta/qwen_32/', 'final/')
df['RoBERTa-SLM-SEL'] = prepare_plm_file(path2+'select/roberta/roberta/qwen_32/', 'final/')

df = prepare_df(df)

latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)


temp_df = df.loc['Mean', :]
vals = [temp_df[::3].to_list(), temp_df[1::3].to_list(), temp_df[2::3].to_list()]
names = temp_df[::3].index.to_list()
plot_ordered_bars(vals, names, segment_labels=['Original', 'UMC', 'SELECT'], title='')