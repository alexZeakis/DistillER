import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import prepare_train_file, prepare_df

import matplotlib.pyplot as plt

def plot_model_lines(series, output_pdf="slm_percentage.pdf"):
    """
    Plots a line chart from a pd.Series with model-percentage index.
    
    Args:
        series (pd.Series): Series with index like 'Model-5%', 'Model-10%', etc.
        output_pdf (str): Path to save the plot as PDF.
    """
    # Extract models and percentages
    models = sorted(set(idx.split('-')[0] for idx in series.index))
    percentages = sorted(set(idx.split('-')[1] for idx in series.index), key=lambda x: int(x.replace('%','')))
    
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size': 14})  # Set font size

    for model in models:
        values = [series[f"{model}-{p}"] for p in percentages]
        plt.plot(percentages, values, marker='o', label=model)
    
    plt.xlabel("Training Dataset Percentage")
    plt.ylabel("F1-score")
    # plt.title("Model Values per Percentage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.show()
    plt.close()

path = '../../../log/matching/annotate/slm/llm/'
path2 = '../../../log/matching/annotate/slm/ground/'
gt_path = '../../../log/matching/annotate/llm/ground/partial/'
df = pd.DataFrame()
df['SMiniLM-Ground'] = prepare_train_file(path2+'sminilm/', gt_path)
# df['SMiniLM-8b'] = prepare_train_file(path+'sminilm/llama_8/', gt_path)
df['SMiniLM-70b'] = prepare_train_file(path+'sminilm/llama_70/', gt_path)
# df['SMiniLM-14'] = prepare_train_file(path+'sminilm/qwen_14/', gt_path)
df['SMiniLM-32'] = prepare_train_file(path+'sminilm/qwen_32/', gt_path)

df['RoBERTa-Ground'] = prepare_train_file(path2+'roberta/', gt_path)
# df['RoBERTa-8b'] = prepare_train_file(path+'roberta/llama_8/', gt_path)
df['RoBERTa-70b'] = prepare_train_file(path+'roberta/llama_70/', gt_path)
# df['RoBERTa-14'] = prepare_train_file(path+'roberta/qwen_14/', gt_path)
df['RoBERTa-32'] = prepare_train_file(path+'roberta/qwen_32/', gt_path)

df = prepare_df(df)
latex_code = df.to_latex(index=True, escape=False, multirow=False)


path = '../../../log/matching/annotate/slm/size/'
df = pd.DataFrame()
df['SMiniLM-5%'] = prepare_train_file(path+'sminilm/0.95/', gt_path)
df['SMiniLM-10%'] = prepare_train_file(path+'sminilm/0.90/', gt_path)
df['SMiniLM-20%'] = prepare_train_file(path+'sminilm/0.80/', gt_path)
df['RoBERTa-5%'] = prepare_train_file(path+'roberta/0.95/', gt_path)
df['RoBERTa-10%'] = prepare_train_file(path+'roberta/0.90/', gt_path)
df['RoBERTa-20%'] = prepare_train_file(path+'roberta/0.80/', gt_path)

df = prepare_df(df)
latex_code_2 = df.to_latex(index=True, escape=False, multirow=False)

plot_model_lines(df.loc['Mean', :])