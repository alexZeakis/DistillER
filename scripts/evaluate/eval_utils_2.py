import re
import os
import json
import pandas as pd

def evaluate(true, preds):
    recall = len(true & preds) / len(true) if len(preds) != 0 else 0
    precision = len(true & preds) / len(preds) if len(preds) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return recall, precision, f1

def find_integers(input_string):
    integers = re.findall(r'-?\b\d+\b', input_string)
    integers = [int(num) for num in integers]
    return integers

def find_integers_in_brackets(text):
    integers = re.findall(r'\[(\d+)\]', text)
    integers = [int(num) for num in integers]
    if len(integers) == 0:
        integers = re.findall(r'#.*?(\d)', text)
        integers = [int(num) for num in integers]
    return integers

def find_integers_in_brackets_orca(text):
    pattern = r'### Final answer:\s*(?:\{[^}]+\}\s*)?\[(\d+)\]'
    integers = re.findall(pattern, text)
    integers = [int(num) for num in integers]
    if len(integers) == 0: #Most likely answered None or paraphrased this one.
        integers = [0]
    return integers

def find_integer(response, path):
    if 'llama' in path:
        t_preds = find_integers_in_brackets(response)
    elif 'mistral' in path:
        t_preds = find_integers_in_brackets(response)
        if len(t_preds)==0:
            t_preds = find_integers(response) 
    elif 'orca' in path:
        # t_preds = find_integers_in_brackets_orca(response)
        t_preds = find_integers_in_brackets(response)
    else:
        t_preds = find_integers_in_brackets(response)
        if len(t_preds)==0:
            t_preds = find_integers(response)
            
    # if len(t_preds) == 0:
        # print(response)
    return t_preds

def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def find_preds_file(response_file, weights=None):
    true, preds = [], []
    total_time = 0
    total_size = 0
    
    counter_good, counter_bad = 0, 0
    with open(response_file) as f:
        j = json.load(f)
        
        m = j['settings']['model']
        for line in j['responses']:
            qid = line['query_id']
            
            t_preds = find_integer(line['response'], m)
                
            
            if len(t_preds)==0:
                # pred = -1 # llm gave no prediction
                pred = 0 #TODO: Merged N3 to N1
            elif len(t_preds) == 1:
                pred = t_preds[0] # only one prediction, desired behavior
            else:
                pred = t_preds[-1] # probably the last number is the predicted one
                # pred = t_preds[0] # only one prediction, desired behavior
            
            true_val = None
            if line['ground_truth'] != -1: #M1 & M2
                true.append((qid, line['answer']))
                true_val = line['answer']
            else: #TODO M3
                # true.append((qid, -1))
                true.append((qid, 0)) # Merged M3 to M1
                true_val = 0
                
            # print(qid, pred, t_preds)

            # if pred > 0: # N1 only
            if pred >= 0: # N1 & N2
                if weights is not None:
                    if type(weights)==dict:
                        w = weights[m]
                    else: # constant, i.e. w=1
                        w = weights
                else:
                    w = 0
                # w = weights[m] if weights is not None else 0
                preds.append((qid, pred, w))
                
            if pred == true_val:    
                counter_good += 1
            else:
                counter_bad += 1
                print(qid, pred, true_val)
                print(line['response'])
            total_time += line['time']
            total_size += len(line['response'])
        total_size /= len(j['responses'])

    print("Good: {}, Bad: {}".format(counter_good, counter_bad))
    return {'seed': j['settings'].get('seed'),
            'dataset': j['settings']['dataset'],
            'model': j['settings']['model'],
            'preds': preds, 'true': true,
            'time': total_time, 'size': total_size}
    


def calc_scores_weighted_vote(dir_path, weights=None):
    
    # out = open('temp_2.txt', 'w')
    scores = []
    total_logs = {}     
    files = find_json_files(dir_path)
    total_size = {}
    for file in files:
        log = find_preds_file(file, weights=weights)
        if log['dataset'] not in total_logs:
            total_logs[log['dataset']] = []
            total_size[log['dataset']] = 0
        total_logs[log['dataset']].append(log)
        total_size[log['dataset']] += log['size']
    for k in total_size.keys():
        total_size[k] = total_size[k] / len(total_logs[k])
        
    for dataset, log_list in total_logs.items():
        total_preds = {}
        for log in log_list:
            temp_true = log['true']
            for k, v, w in log['preds']:
                if k not in total_preds:
                    total_preds[k] = []
                total_preds[k].append((v, w))
                
        preds = []
        for k, weighted_votes in total_preds.items():
            if weights is None:
                for pred, weight in weighted_votes:
                    preds.append((k, pred))
            else:
                # out.write('{},{},{}\n'.format(dataset, k, weighted_votes))
                vote_count = {}
                for pred, weight in weighted_votes:
                    if pred not in vote_count:
                        vote_count[pred] = 0
                    vote_count[pred] += weight
                
                # Find predictions with the highest weighted sum
                max_weight = max(vote_count.values())
                tied_preds = [pred for pred, weight in vote_count.items() if weight == max_weight]

                if len(tied_preds) == 1: #no ties
                    preds.append((k, tied_preds[0]))
        
        temp_true = set(temp_true)
        
        # print(len(true))
        preds = set(preds)
        recall, precision, f1 = evaluate(temp_true, preds)
        
        log = {}
        log['dataset'] = dataset
        log['recall'] = recall
        log['precision'] = precision
        log['f1'] = f1
        log['size'] = total_size[dataset]
        # log['time'] = total_time
        scores.append(log)
        
    scores = pd.DataFrame(scores)
    return scores

def transform_series(scores, cols=None):
    scores = scores[['dataset', 'f1']].set_index('dataset')
    if cols is None:
        scores = scores.T.values[0]
    else:
        # valid_cols = [col for col in cols if col in scores.index]
        # scores = scores.loc[valid_cols].T.values[0]
        scores = scores.reindex(cols, fill_value=0).T.values[0]
    return scores

# Function to apply styles
def highlight_values(series):
    # Get indices of the highest and second-highest values
    sorted_vals = series.drop_duplicates().sort_values(ascending=False)
    first_val = sorted_vals.iloc[0]
    if sorted_vals.shape[0] > 1:
        second_val = sorted_vals.iloc[1]
    
    res = []
    for index, row in series.items():
        if row == first_val:
            res += [fr'\textbf{{{row}}}']
        elif second_val is not None and row == second_val:
            res += [fr'\underline{{{row}}}']
        else:
            res += [fr'{row}']
    return res

# Apply the styling for LaTeX
def latex_style(df):
    columns_to_style = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'Mean']
    styled_df = df.copy()
    for col in columns_to_style:
        if col in df.columns:
            styled_df[col] = highlight_values(df[col])
    return styled_df

def transform_series_2(scores, extras={}):
    x = scores[['dataset', 'f1']].set_index('dataset')
    x.loc['Mean', :] = x.mean()
    x.loc['Rsp. Size', :] = scores['size'].mean()
    x = x.to_dict()['f1']
    for k,v in extras.items():
        x[k] = v
    return x
