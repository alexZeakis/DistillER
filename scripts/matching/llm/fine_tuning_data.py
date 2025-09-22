import os
import argparse
import json

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
    parser.add_argument('--explanations', action='store_true', 
                        help='Add explanations if flag is set')
    
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
    
            if args.mode == 'train':
                answer = "[{}]".format(p[field])
                if args.explanations:
                    answer = "Answer:[{}]. {}".format(p[field], p['explanation'])
                conv = [
                    {"from":"human", "value": p['prompt']},
                    {"from":"gpt", "value":f"{answer}"}
                    ]
            elif args.mode == 'test':
                conv = [
                    {"from":"human", "value": p['prompt']},
                    ]
                
            d = {'id': '{}_{}_{}'.format(dataset, seed, p['query_id']),
                 'conversations': conv,
                 'true': p['ground_answer'], 'gt': p['ground_truth']
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