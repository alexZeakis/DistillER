import os
import argparse
import json
import re
import pandas as pd

def find_json(text):
    
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

def find_enhanced_json(text):
    
    pattern = '\{\s*"answer"\s*:\s*".*?"\s*,\s*"explanation"\s*:\s*".*?"\s*\,\s*"confidence"\s*:\s*".*?"\s*\}'
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
                return {'answer': last_match, 'explanation': '', 'confidence': 1.0}
            
    pattern = r'\[\d+\]'
    matches = re.findall(pattern, text, re.DOTALL)
    last_match = matches[-1] if matches else None
    if last_match is not None: #successful structured output
        return {'answer': last_match, 'explanation': '', 'confidence': 1.0}
    return {}
    

def clean_labels(responses, confidence_flag=False, confidence_threshold=0.0):
    labels = {}
    confidence = {}
    for res in responses:
        if res['explanation'] is None: #error
            if res['answer'] is None: #timeout
                answer = {}
            else: #error in Structured output
                if confidence_flag:
                    answer = find_enhanced_json(res['answer'])
                else:
                    answer = find_json(res['answer'])
                
                if len(answer) > 0:
                    if 'confidence' in answer:
                        conf = answer['confidence']
                    else: # not confidence_flag
                        conf = 1.0
                    answer = answer['answer']
        else:
            answer = res['answer']
            if 'confidence' in res:
                conf = res['confidence']
            else: # not confidence_flag
                conf = 1.0
        
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                answer = 0
        else:
            answer = 0
            conf = 1.0
            
        labels[res['query_id']] = answer
        confidence[res['query_id']] = float(conf)
    return labels, confidence
    

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    parser.add_argument('--in_files',  nargs='+', required=True, help='List of response files.')
    parser.add_argument('--out_file', type=str, required=True, help='Output file')
    
    args = parser.parse_args()
    
    total_labels = pd.DataFrame()
    total_conf = pd.DataFrame()
    for nof, file in enumerate(args.in_files):
        with open(file) as f:
            original_labels = json.load(f)
            
        labels, confidence = clean_labels(original_labels['responses'], False, 0.0)
        
        total_labels[nof] = pd.Series(labels)
        total_conf[nof] = pd.Series(confidence)

    total_labels = total_labels.mode(axis=1)[0].astype(int)
    total_conf = total_conf.mean(axis=1)
    
    for label in original_labels['responses']:
        qid = label['query_id']
        label['answer'] = '[{}]'.format(total_labels[qid])
        label['confidence'] = total_conf[qid]
        
        
    path2 = args.out_file
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    with open(path2, 'w') as f:
        f.write(json.dumps(original_labels, indent=4))        