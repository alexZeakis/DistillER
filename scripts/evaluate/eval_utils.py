import json
import pandas as pd
import re
import os
import matplotlib.pyplot as plt


def compute_f1_score(y_true, y_pred):
    """
    Compute the F1 score for binary classification without using sklearn.

    Parameters:
        y_true (list of int): Ground truth labels (0 or 1).
        y_pred (list of int): Predicted labels (0 or 1).

    Returns:
        float: F1 score.
    """
    assert len(y_true) == len(y_pred), "Length of true and predicted labels must match"

    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0  # Avoid division by zero

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    print(precision, recall)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calc_f1(true, preds):
    if len(true) > 0:
        recall = len(true & preds) / len(true)
    if len(preds) > 0:
        precision = len(true & preds) / len(preds)
    if len(true) > 0:
        if precision == recall == 0:
            return 0
        return 2 * precision * recall / (precision+recall)




def get_scores_from_ft_json(path, measure=calc_f1):
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
            
    score = measure(ground_results, predictions)
    
    # return ground_results, predictions
    return dataset, score


def get_scores_from_json(path, measure=calc_f1):
    j = json.load(open(path))
    dataset = j['settings']['dataset']
    
    ground_results = set()
    predictions = set()
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
        # if res['ground_answer'] == -1: #TODO: Check for certainty
        #     ground_answer = 0
        
        ground_results.add((res['query_id'], res['ground_answer']))
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            predictions.add((res['query_id'], answer))
            
    score = measure(ground_results, predictions)
    
    return dataset, score








def prepare_pt_file(path, measure=calc_f1):
    temp_scores = {}
    for file in os.listdir(path):
        if 'responses' not in file:
            continue
        dataset, score = get_scores_from_json(path+file, measure=measure)
        temp_scores[dataset] = score
    return pd.Series(temp_scores)

def prepare_ft_file(path, measure=calc_f1):
    temp_scores = {}
    for file in os.listdir(path):
        if 'responses' not in file:
            continue
        dataset, score = get_scores_from_ft_json(path+file, measure=measure)
        temp_scores[dataset] = score
    return pd.Series(temp_scores)

def prepare_umc_file(path, measure='F1'):
    temp_scores = {}
    for file in os.listdir(path):
        df = pd.read_csv(path+file)
        temp_scores[file.split('.')[0]] = df.iloc[-1][measure]
    # total_df_test['UMC'] = pd.Series(temp_scores)
    return pd.Series(temp_scores)

# def prepare_plm_file(path):
#     scores = {}
#     with open(path+'log/matching_supervised_dynamic.txt') as f:
#         for line in f:
#             j = json.loads(line)
#             scores[j['data_name']] = j['f1']
#     return pd.Series(scores)

def prepare_plm_file(path, rdir='log/', measure=compute_f1_score):
    scores = {f'D{i}': {'preds': [], 'true': []} for i in range(2,10)}
    for file in os.listdir(path+rdir):
        if not file.endswith('csv'):
            continue
        df = pd.read_csv(path + rdir + file)
    
        for index, row in df.iterrows():
            case = 'D{}'.format(row['datasets'])
            scores[case]['preds'].append(row['predictions'])
            scores[case]['true'].append(row['labels'])

    results = {}
    for key, val in scores.items():
        results[key] = float(measure(val['true'], val['preds']))
    return pd.Series(results)



def prepare_plm_train_file(path, measure=compute_f1_score):
    df = pd.read_csv(path + 'log/total_predictions.csv')
    scores = {f'D{i}': {'preds': [], 'true': []} for i in range(2,10)}
    for index, row in df.iterrows():
        case = 'D{}'.format(row['datasets'])
        scores[case]['preds'].append(row['predictions'])
        scores[case]['true'].append(row['labels'])

    results = {}
    for key, val in scores.items():
        results[key] = float(measure(val['true'], val['preds']))
    return pd.Series(results)
        




def prepare_plm_file_cardinality(path):
    values = []
    for file in os.listdir(path+'log/'):
        if not file.endswith('csv'):
            continue
        
        df = pd.read_csv(path + 'log/' + file)
        df = df.loc[df.predictions == 1]
        df = df.groupby('left_ID')['right_ID'].count()
        values += df.values.tolist()
        
    s = pd.Series(values)
    
    d = {}
    d['50%'] = s.quantile(0.5)
    d['95%'] = s.quantile(0.95)
    d['max'] = s.max()
    return pd.Series(d)


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

    
def get_inference_time(path):    
    times = {}
    for file in os.listdir(path):
        if 'responses' not in file:
            continue
        with open(path+file) as f:
            j = json.load(f)
            case = j['settings']['dataset']
            times[case] = 0
            for res in j['responses']:
                times[case] += res['time']
    return pd.Series(times)

def get_plm_training_time(path):
    times = 0
    with open(path+'log/matching_supervised_dynamic.txt') as f:
        for line in f:
            j = json.loads(line)
            times += j['training_time']
    return times

def get_plm_testing_time(path):
    times = 0
    with open(path+'log/matching_supervised_dynamic.txt') as f:
        for line in f:
            j = json.loads(line)
            times += j['testing_time']
    return times
    


def create_plot_comparison(total_df_test):
    colors = {
        "UMC": "blue",
        "Pretrained": "orange",
        "GT": "green",
        "Llama3.1:70b": "red",
        "Llama3.1:8b": "purple",
        "Qwen2.5:32b": "brown"
    }

    df_reset = total_df_test.reset_index().rename(columns={'index': 'Dataset'})
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
    
def prepare_df(df):
    order = [f'D{no}' for no in range(2,10)]
    df = df.loc[order]
    df.loc['Mean', :] = df.mean()
    df = df.round(2)
    return df