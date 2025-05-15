import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt


def prepare_plm_file(path):
    scores = {}
    with open(path+'log/matching_supervised_dynamic.txt') as f:
        for line in f:
            j = json.loads(line)
            scores[j['data_name']] = j['f1']
    return pd.Series(scores)
        
def get_scores_from_ft_json(path):
    j = json.load(open(path))
    dataset = j['settings']['dataset']
    
    ground_results = set()
    predictions = set()
    for res in j['responses']:
        # print(res)

        answer = find_integer(res['response'])
        if len(answer) > 0:
            answer = answer['answer']
        
        ground_results.add((res['query_id'], res['answer']))
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            predictions.add((res['query_id'], answer))
        # if answer != res['answer']:
        #     print("Response: {}\nShortResponse: {}\nAnswer: {}\n".format(res['response'], answer, res['answer']))
            
    f1 = calc_f1(ground_results, predictions)
    
    # return ground_results, predictions
    return dataset, f1

def get_scores_from_json(path):
    j = json.load(open(path))
    dataset = j['settings']['dataset']
    
    ground_results = set()
    predictions = set()
    for res in j['responses']:
        # print(res)
        # if res['query_id'] == 2054:
        #     # print(res['answer'], res['explanation'])
        #     print(res)

        if res['explanation'] is None: #error
            if res['answer'] is None: #timeout
                answer = {}
            else: #error in Structured output
                answer = find_json(res['answer'])
                if len(answer) > 0:
                    answer = answer['answer']
        else:
            answer = res['answer']
        # if res['ground_answer'] == -1: #TODO: Check for certainty
        #     ground_answer = 0
        
        ground_results.add((res['query_id'], res['ground_answer']))
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            predictions.add((res['query_id'], answer))
            
    f1 = calc_f1(ground_results, predictions)
    
    # return ground_results, predictions
    return dataset, f1

def find_integer(text):
    pattern = r'\[\d+\]'
    matches = re.findall(pattern, text, re.DOTALL)
    last_match = matches[0] if matches else None
    if last_match is not None: #successful structured output
        return {'answer': last_match, 'explanation': ''}
    return {}

def find_json(text):
    # print(text)
    
    # pattern = r'({"answer":.*?"explanation".*?})'
    pattern = '\{\s*"answer"\s*:\s*".*?"\s*,\s*"explanation"\s*:\s*".*?"\s*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    last_match = matches[-1] if matches else None
    if last_match is not None: #successful structured output
        try:
            return json.loads(last_match)
        except:
            pattern = '\"answer"\s*:\s*".*?"\s*,'
            matches = re.findall(pattern, text, re.DOTALL)
            last_match = matches[-1] if matches else None
            if last_match is not None: #successful structured output
                return {'answer': last_match, 'explanation': ''}
            
    pattern = r'\[\d+\]'
    matches = re.findall(pattern, text, re.DOTALL)
    last_match = matches[-1] if matches else None
    if last_match is not None: #successful structured output
        return {'answer': last_match, 'explanation': ''}
    return {}


def calc_f1(true, preds):
    if len(true) > 0:
        recall = len(true & preds) / len(true)
    if len(preds) > 0:
        precision = len(true & preds) / len(preds)
    if len(true) > 0:
        return 2 * precision * recall / (precision+recall)
    
    
def prepare_pt_file(path):
    temp_scores = {}
    for file in os.listdir(path):
        if 'responses' not in file:
            continue
        dataset, f1 = get_scores_from_json(path+file)
        temp_scores[dataset] = f1
    return pd.Series(temp_scores)

def prepare_ft_file(path):
    temp_scores = {}
    for file in os.listdir(path):
        if 'responses' not in file:
            continue
        dataset, f1 = get_scores_from_ft_json(path+file)
        temp_scores[dataset] = f1
    return pd.Series(temp_scores)
    

order = [f'D{no}' for no in range(2,10)]

total_df = pd.DataFrame()

temp_scores = {}
path = '../../log/matching/baselines/umc/'
for file in os.listdir(path):
    df = pd.read_csv(path+file)
    temp_scores[file.split('.')[0]] = df.iloc[-1]['F1']
total_df['UMC'] = pd.Series(temp_scores)

path = '../../log/matching/'
total_df['Pretrained'] = prepare_pt_file(path+'baselines/pretrained/')    
total_df['GT'] = prepare_ft_file(path+'finetuning/ground/test_responses/')
# total_df['Llama3:70b'] = prepare_ft_file(path+'finetuning/noisy/test_responses/')
total_df['Llama3.1:8b'] = prepare_ft_file(path+'finetuning/noisy_3/test_responses/')
total_df['Qwen2.5:32b'] = prepare_ft_file(path+'finetuning/noisy_4/test_responses/')
total_df['Llama3.1:70b'] = prepare_ft_file(path+'finetuning/noisy_6/test_responses/')

total_df = total_df.loc[order]

# total_df[['UMC', 'Pretrained', 'GT', 'Llama3.1:8b', 'Qwen2.5:32b']].plot.line()

colors = {
    "UMC": "blue",
    "Pretrained": "orange",
    "GT": "green",
    "Llama3.1:70b": "red",
    "Llama3.1:8b": "purple",
    "Qwen2.5:32b": "brown"
}

df_reset = total_df.reset_index().rename(columns={'index': 'Dataset'})
ax = None
for column in colors.keys():
    ax = df_reset.plot.scatter(
        x='Dataset',
        y=column,
        label=column,
        ax=ax,
        color=colors.get(column, 'gray'),
        s=60  # marker size
    )
ax.set_title("Scatter Plot of Methods per Dataset")
ax.set_xlabel("Dataset")
ax.set_ylabel("F1")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


total_df.loc['Mean', :] = total_df.mean()
total_df = total_df.round(2)






path = '../../log/matching/finetuning/'
total_df_2 = pd.DataFrame()
# total_df_2['Llama3:70b'] = prepare_pt_file(path+'noisy/partial_responses/')
# total_df_2['Llama3.3:70b'] = prepare_pt_file(path+'noisy_2/partial_responses/')
total_df_2['Llama3.1:8b'] = prepare_pt_file(path+'noisy_3/partial_responses/')
# total_df_2['Llama3.1:8b(Groq)'] = prepare_pt_file(path+'noisy_5/partial_responses/')
total_df_2['Qwen2.5:32b'] = prepare_pt_file(path+'noisy_4/partial_responses/')
total_df_2['Llama3.1:70b'] = prepare_pt_file(path+'noisy_6/partial_responses/')

total_df_2 = total_df_2.loc[order]
total_df_2.loc['Mean', :] = total_df_2.mean()
total_df_2 = total_df_2.round(2)


path = '../../log/matching/plm/'
total_df_3 = pd.DataFrame()
total_df_3['GT'] = prepare_plm_file(path+'ground/')
total_df_3['Llama3.1:8b'] = prepare_plm_file(path+'noisy_3/')
total_df_3['Qwen2.5:32b'] = prepare_plm_file(path+'noisy_4/')
total_df_3['Llama3.1:70b'] = prepare_plm_file(path+'noisy_6/')
total_df_3['GT_DS2:1'] = prepare_plm_file(path+'ground_ds/')
total_df_3['Llama3.1:70b_DS2:1'] = prepare_plm_file(path+'noisy_6_ds/')

total_df_3 = total_df_3.loc[order]
total_df_3.loc['Mean', :] = total_df_3.mean()
total_df_3 = total_df_3.round(2)


path = '../../log/matching/'
total_df_4 = pd.DataFrame()
total_df_4['Llama3.1:8b'] = prepare_ft_file(path+'finetuning/noisy_3/test_responses/')
total_df_4['Llama3.1:8b - LR_θ=0.9'] = prepare_ft_file(path+'confidence/noisy/test_responses/')

total_df_4 = total_df_4.loc[order]
total_df_4.loc['Mean', :] = total_df_4.mean()
total_df_4 = total_df_4.round(2)








     

# temp_scores = {}
# path = '../../log/matching/finetuning/noisy/partial_responses/'
# for file in os.listdir(path):
#     if 'responses' not in file:
#         continue
    
#     if 'D4' not in file:
#         continue
#     ground_results, predictions = get_scores_from_json(path+file)
    
#     # ground_results = {k:v for (k,v) in ground_results}
#     # predictions = {k:v for (k,v) in predictions}
#     # for k, v in predictions.items():
#     #     v2 = ground_results[k]
#     #     if v == v2:
#     #         continue
#     #     print('For {}, prediction: {}, true: {}'.format(k, v, v2))
    
#     # dataset, f1 = get_scores_from_json(path+file)
# #     temp_scores[dataset] = f1
# # total_df['Noisy-Train'] = pd.Series(temp_scores)