import pandas as pd
import argparse
import os
import json
import re

# def keep(row, cleaned):
#     if row['predictions'] == 0:
#         return True
#     else:
#         if (int(row['left_ID']), int(row['right_ID'])) in cleaned:
#             return True
#     return False

def keep(row, cleaned, cleaned_k):
    if row['predictions'] == 0:
        return 0
    else:
        if int(row['left_ID']) not in cleaned_k: # 1 option
            return 1
        if (int(row['left_ID']), int(row['right_ID'])) in cleaned:
            return 1
    return 0

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

def get_predictions(path, prompts):
    j = json.load(open(path))
    
    predictions = []
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
        
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            
            if answer > 0:
                answer = prompts[res['query_id']][answer-1]
            else:
                answer = -1
                
            predictions.append((res['query_id'], answer))
            
    return set(predictions)

if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--input_file', type=str, required=True, help='Input file')
    parser.add_argument('--cleaned_file', type=str, required=True, help='Cleaned of UMC file')
    parser.add_argument('--prompts_file', type=str, required=True, help='Prompts file')
    parser.add_argument('--out_file', type=str, required=True, help='Out file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    prompts = json.load(open(args.prompts_file))
    prompts = {prompt['query_id']: prompt['options'] for prompt in prompts['prompts']}
    
    cleaned = get_predictions(args.cleaned_file, prompts)
    cleaned_k = set([c[0] for c in cleaned])
    
    edges = pd.read_csv(args.input_file, index_col=0)
    # x = edges.shape
    # edges = edges.loc[edges.apply(lambda x: keep(x, cleaned), axis=1)]
    # y = edges.shape
    edges.predictions = edges.apply(lambda x: keep(x, cleaned, cleaned_k), axis=1)
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    edges.to_csv(args.out_file, header=True, index=True)
    
    