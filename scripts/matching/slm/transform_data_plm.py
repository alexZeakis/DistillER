import os
import pandas as pd
import argparse
import json
from utils import settings
import traceback
import re
from sklearn.model_selection import train_test_split
import random



def downsample_majority(data, ratio_N=2, seed=42):
    random.seed(seed)

    # Separate data by class
    majority = [item for item in data if item[4] == 0]
    minority = [item for item in data if item[4] == 1]

    # Compute how many majority samples we need
    target_majority_count = ratio_N * len(minority)

    # Downsample majority class if needed
    if len(majority) > target_majority_count:
        majority = random.sample(majority, target_majority_count)

    # Combine and shuffle
    balanced = majority + minority
    random.shuffle(balanced)

    return balanced


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

def get_scores_from_ft_json(path):
    j = json.load(open(path))
    
    predictions = {}
    for res in j['responses']:

        answer = find_integer(res['response'])
        if len(answer) > 0:
            answer = answer['answer']
        
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            # predictions.add((res['query_id'], answer))
            predictions[res['query_id']] = answer
    return predictions
    
def get_scores_from_json(path):
    j = json.load(open(path))
    
    predictions = {}
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
            predictions[res['query_id']] = [answer]
            
    return predictions

def get_scores_from_noisy_json(path):
    j = json.load(open(path))
    predictions = {}
    for res in j['prompts']:
        qid = res['query_id']
        if type(qid) != int:
            qid = int(qid.split('_')[0])
        if qid not in predictions:
            predictions[qid] = []
        # predictions[res['query_id']] = res['noise_answer']
        predictions[qid].append(res['noise_answer'])
    return predictions


def serialize(row, serialization='DITTO'):
    desc = ''
    if serialization == 'llm':
        return row['description']
        
    for col, val in row.items():
        if col.startswith('http'):
            col = col.split('/')[-1]
        if pd.isna(val):
            continue
        if serialization == 'DITTO':
            desc += f' [COL] {col} [VAL] {val}'
        elif serialization == 'schema_agnostic':
            desc += f' {val}'  
    desc = re.sub(r"[{}]", "", desc)  
    return desc


def shuffle_on_seed(df, seed):
    df = df.sample(frac=1, random_state=seed) # shuffle for position bias
    return df
    
def find_answer_on_shuffle(df, answer):
    if answer is None:
        return None
    
    if answer in df.index:
        position = df.index.get_loc(answer)
        true_position = position+1
    else:
        true_position = 0
    return true_position

def prepare_pairs(row1, df2, serialization='DITTO'):
    
    pairs = []
    entity_description = serialize(temp_row1, serialization)
    
    for index, row in df2.iterrows():
        candidate_description = serialize(row, serialization)
        pairs.append((entity_description, candidate_description, index))
    
    return pairs

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_dir', type=str, required=True, help='Out directory')
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--sample_file', type=str, required=True, help='Sample file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode (Train or Test)')
    
    parser.add_argument('--seed', type=int, default=1924, required=False, help='Seed for shuffling')
    parser.add_argument('--serialization', type=str, choices=['DITTO', 'schema_agnostic', 'llm'],
                        required=True, help='Serialization Strategy')
    parser.add_argument('--response_file', type=str, required=False, help='Response file with noisy predictions')
    parser.add_argument('--percentage', type=float, default=0.8, required=False, help='Percentage of splitting')
    parser.add_argument('--downsampling', type=int, default=None, required=False, help='Ratio to downsampling (N:1)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    dataset = args.dataset
    serialization=args.serialization
    seed=args.seed
    
    # Print the values of the arguments
    print(args)
    
    datasets = settings[dataset]
    # files = os.listdir(args.emb_dir+dataset)
    file1 = '{}{}/{}.csv'.format(args.in_dir, dataset, datasets['D1'])
    file2 = '{}{}/{}.csv'.format(args.in_dir, dataset, datasets['D2'])
    
    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)
    
    sample_df = pd.read_csv(args.sample_file)
    print(df1.shape, df2.shape, sample_df.shape)
    
    total_cands = sample_df.groupby('D1')['D2'].apply(list)
    ground_truth = dict(sample_df[['D1', 'True']].drop_duplicates().values)
    
    
    predictions = None
    if args.response_file is not None:
        if 'partial/' in args.response_file:
            predictions = get_scores_from_json(args.response_file)
        elif 'partial_noisy/' in args.response_file:
            predictions = get_scores_from_noisy_json(args.response_file)
        elif 'partial_responses/' in args.response_file:
            predictions = get_scores_from_json(args.response_file)            
        else:
            raise ValueError("No valid file is given.")
            
    total_pairs = []
    for no, (key, cands) in enumerate(total_cands.items()):
        if no % 50 == 0:
            print('Query {}/{}\r'.format(no, len(total_cands)), end='')
        
        temp_row1 = df1.loc[key]
        temp_df2 = df2.loc[cands]
        temp_df2 = shuffle_on_seed(temp_df2, seed)
        true_answer = int(ground_truth.get(key, -1))
        if predictions is None:
            answers = [find_answer_on_shuffle(temp_df2, true_answer)]
        else:
            if key not in predictions: #only for PLM where we split the training data
                continue
            answers = predictions[key]
            
        pairs = prepare_pairs(temp_row1, temp_df2, serialization=serialization)
        
        for answer in answers: # only for PLM, where possible multiple positives
            for no, pair in enumerate(pairs):
                label = (answer-1==no) * 1
                x = (key, pair[2], pair[0], pair[1], label)
                total_pairs.append(x)


    columns = ['Left_ID', 'Right_ID', 'Left_Text', 'Right_Text', 'Label']

    if args.mode == 'train':
        if args.downsampling is not None:
            total_pairs = downsample_majority(total_pairs, ratio_N=args.downsampling)
        
        labels = [label for _, _,_, _, label in total_pairs]
        train_data, valid_data = train_test_split(
            total_pairs,
            test_size=1-args.percentage,
            stratify=labels,
            random_state=1924  # for reproducibility
        )
        
        path2 = args.out_dir+"train.csv"
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        train = pd.DataFrame(train_data, columns=columns)
        train['Dataset'] = dataset
        train.to_csv(path2, header=True, index=False)
        
        path2 = args.out_dir+"valid.csv"
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        valid = pd.DataFrame(valid_data, columns=columns)
        valid['Dataset'] = dataset
        valid.to_csv(path2, header=True, index=False)
    else:
        path2 = args.out_dir+"test.csv"
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        test = pd.DataFrame(total_pairs, columns=columns)
        test['Dataset'] = dataset
        test.to_csv(path2, header=True, index=False)