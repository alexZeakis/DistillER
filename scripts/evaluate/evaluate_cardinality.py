import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from collections import Counter


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
    
    c = Counter()
    for res in j['responses']:

        if res['explanation'] is None: #error
            if res['answer'] is None: #timeout
                answer = {}
            else: #error in Structured output
                answer = find_json(res['answer'])
                if len(answer) > 0:
                    answer = answer['answer']
        else:
            answer = res['answer']
        
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            c.update([res['query_id']])
            
    
    return j['settings']['dataset'], pd.Series(c).describe()

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
        dataset, description = get_scores_from_json(path+file)
        temp_scores[dataset] = description
    return pd.DataFrame(temp_scores)

def prepare_ft_file(path):
    temp_scores = {}
    for file in os.listdir(path):
        if 'responses' not in file:
            continue
        dataset, f1 = get_scores_from_ft_json(path+file)
        temp_scores[dataset] = f1
    return pd.Series(temp_scores)
    

order = [f'D{no}' for no in range(2,10)]

total_df_test = pd.DataFrame()

temp_scores = {}
path = '../../log/matching/baselines/umc/'
for file in os.listdir(path):
    df = pd.read_csv(path+file)
    temp_scores[file.split('.')[0]] = df.iloc[-1]['F1']
total_df_test['UMC'] = pd.Series(temp_scores)

path = '../../log/matching/'
# total_df_test['Pretrained'] = prepare_pt_file(path+'baselines/pretrained/')    
# total_df_test['GT'] = prepare_ft_file(path+'finetuning/ground/test_responses/')
# total_df_test['Llama3.1:8b'] = prepare_ft_file(path+'finetuning/noisy_3/test_responses/')
# total_df_test['Llama3.1:70b'] = prepare_ft_file(path+'finetuning/noisy_6/test_responses/')
# total_df_test['Qwen2.5:7b'] = prepare_ft_file(path+'finetuning/noisy_7/test_responses/')
# total_df_test['Qwen2.5:32b'] = prepare_ft_file(path+'finetuning/noisy_4/test_responses/')

# total_df_test = total_df_test.loc[order]
# total_df_test.loc['Mean', :] = total_df_test.mean()
# total_df_test = total_df_test.round(2)



path = '../../log/matching/finetuning/'
# total_df_train = pd.DataFrame()
df = prepare_pt_file(path+'noisy_3/partial_responses/')
# total_df_train['Llama3.1:70b'] = prepare_pt_file(path+'noisy_6/partial_responses/')
# total_df_train['Qwen2.5:7b'] = prepare_pt_file(path+'noisy_7/partial_responses/')
# total_df_train['Qwen2.5:32b'] = prepare_pt_file(path+'noisy_4/partial_responses/')

# total_df_train = total_df_train.loc[order]
# total_df_train.loc['Mean', :] = total_df_train.mean()
# total_df_train = total_df_train.round(2)




