import os
import argparse
import json
import re

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--input_files', nargs='+', type=str, required=True, help='List of input files')
    parser.add_argument('--out_file', type=str, required=True, help='Output file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], 
                        help='Mode of data.')
    parser.add_argument('--size', required=False, default=1.0, 
                        type=float, help='Percentage of data to use.')
    parser.add_argument('--field', required=False, default="ground_answer", 
                        type=str, help='Field with the correct answer.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    print(args)
    prompts = []    
    for file in args.input_files:
        with open(file) as f:
            j = json.load(f)
            
        dataset = j['settings']['dataset']    
        seed = j['settings']['seed'] 
        
        size = int(len(j['prompts']) * args.size)
        original_prompts = j['prompts'][:size]
        field = args.field        
        
        for no, p in enumerate(original_prompts):
            if no % 50 == 0:
                print('Query {}/{}\r'.format(no, len(original_prompts)), end='')
    
            new_task = 'Answer with the corresponding record number surrounded by \"[]\" or \"[0]\" if there is none.'
    
            prompt = p['prompt']
            prompt = re.sub(r'Please respond.*', new_task, prompt, flags=re.DOTALL)
            if args.mode == 'train':
                answer = p[field]
                
                for noc, cand in enumerate(p['options']):
                    if noc == answer:
                        continue
                    
                    d = {'id': '{}_{}_{}_{}'.format(dataset, seed, p['query_id'], noc),
                         'prompt': prompt,
                         'chosen': "[{}]".format(answer),
                         'rejected': "[{}]".format(noc),
                         # 'true': p['ground_answer'], 
                         # 'gt': p['ground_truth']
                         }
                    prompts.append(d)
                    
                # if len(prompts) > 100:
                #     break
            elif args.mode == 'test':
                
                d = {'id': '{}_{}_{}'.format(dataset, seed, p['query_id']),
                     'prompt': prompt,
                     'true': p['ground_answer'], 
                     'gt': p['ground_truth']
                     }
                prompts.append(d)
                
        if args.mode == 'test':
            path2 = args.out_file
            os.makedirs(os.path.dirname(path2), exist_ok=True)
            with open(path2, 'w') as f:
                f.write(json.dumps(prompts, indent=4))        
            prompts = []
    
    if args.mode == 'train':        
        path2 = args.out_file
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        with open(path2, 'w') as f:
            f.write(json.dumps(prompts, indent=4))