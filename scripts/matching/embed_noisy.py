import os
import argparse
import json
import re

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
    
    parser.add_argument('--prompts', type=str, required=True, help='Prompts file')
    parser.add_argument('--labels', type=str, required=True, help='Labels file')
    parser.add_argument('--out_file', type=str, required=True, help='Output file')
    parser.add_argument('--confidence', type=bool, default=False, required=False, help='Confidence flag')
    parser.add_argument('--confidence_threshold', type=float, default=0.0, required=False, help='Confidence Threshold')
    
    args = parser.parse_args()
    
    thres = args.confidence_threshold
    
    with open(args.prompts) as f:
        prompts = json.load(f)
        
    with open(args.labels) as f:
        labels = json.load(f)
        
    labels, confidence = clean_labels(labels['responses'], args.confidence, args.confidence_threshold)

    for prompt in prompts['prompts']:
        prompt['noise_answer'] = labels[prompt['query_id']]
        prompt['confidence'] = confidence[prompt['query_id']]

    # if not confidence_flag, then conf = 1.0 and thres = 0.0, so condition=True
    prompts['prompts'] = [prompt for prompt in prompts['prompts']
                          if abs(prompt['confidence'] - thres) >= 0.000001]
        
    path2 = args.out_file
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    with open(path2, 'w') as f:
        f.write(json.dumps(prompts, indent=4))        