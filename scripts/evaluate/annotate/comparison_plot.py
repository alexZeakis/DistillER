import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eval_utils import get_inference_time, get_plm_time, prepare_df, prepare_train_file

################### COMPARING LLMS ON TRAIN DATA #########################

total_df = pd.DataFrame()

path = '../../../log/matching/annotate/llm/'
path2 = '../../../log/matching/annotate/slm/llm/'
gt_path = '../../../log/matching/annotate/llm/ground/partial/'
df = pd.DataFrame()
df['Llama:70b'] = prepare_train_file(path+'llama_70/', gt_path)
df['Qwen:32b'] = prepare_train_file(path+'qwen_32/', gt_path)
df['Llama:8b'] = prepare_train_file(path+'llama_8/', gt_path)
df['Qwen:14b'] = prepare_train_file(path+'qwen_14/', gt_path)
df['SMiniLM-32'] = prepare_train_file(path2+'sminilm/qwen_32/', gt_path)
df['RoBERTa-32'] = prepare_train_file(path2+'roberta/qwen_32/', gt_path)
df['SMiniLM-70'] = prepare_train_file(path2+'sminilm/llama_70/', gt_path)
df['RoBERTa-70'] = prepare_train_file(path2+'roberta/llama_70/', gt_path)

df = prepare_df(df)

total_df['F1'] = df.loc['Mean']
latex_code = df.to_latex(index=True, escape=False, multirow=False)



df = {}
df['Llama:70b'] = get_inference_time(path+'llama_70/partial_responses/')
df['Qwen:32b'] = get_inference_time(path+'qwen_32/partial_responses/')
df['Llama:8b'] = get_inference_time(path+'llama_8/partial_responses/')
df['Qwen:14b'] = get_inference_time(path+'qwen_14/partial_responses/')
df['SMiniLM-32'] = df['Qwen:32b'] * 0.2 + get_plm_time(path2+'sminilm/qwen_32/')
df['RoBERTa-32'] =  df['Qwen:32b'] * 0.2 + get_plm_time(path2+'roberta/qwen_32/')
df['SMiniLM-70'] = df['Llama:70b'] * 0.2 + get_plm_time(path2+'sminilm/llama_70/')
df['RoBERTa-70'] =  df['Llama:70b'] * 0.2 + get_plm_time(path2+'roberta/llama_70/')
total_df['Time'] = pd.Series(df)

total_df = total_df.reset_index(drop=False)


import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 14})

# Create scatter plot
plt.figure(figsize=(8, 6))
# plt.scatter(total_df["Time"], total_df["F1"], s=60)
plt.scatter(total_df["Time"][:4], total_df["F1"][:4], s=80, marker='*', color='red')
plt.scatter(total_df["Time"][4:], total_df["F1"][4:], s=60, color='blue')


# Add labels for each point
for i, row in total_df.iterrows():
    plt.text(row["Time"]-650, row["F1"]+0.01, row["index"],
             fontsize=14, ha='left', va='bottom', rotation=20)

plt.xlabel("Total Time (sec)")
plt.ylabel("Average F1")
plt.title("Teacher Performance")

plt.xlim(400, 14000)
plt.ylim(0.60, 0.87)

plt.tight_layout()

# --- Save as PDF (optional) ---
# Uncomment the line below to save:
plt.savefig("teacher_performance.pdf", format="pdf")

plt.show()
