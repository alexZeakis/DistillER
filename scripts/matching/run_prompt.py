import os
from time import time
import argparse
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from structured_responses import AnswerSelectSchema, AnswerExplainSchema, \
    AnswerConfidenceSchema, AnswerJustifySchema
import traceback
import httpx

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model', type=str, required=True, help='Name of the LLM model')
    parser.add_argument('--in_file', type=str, required=True, help='Input file')
    parser.add_argument('--out_file', type=str, required=True, help='Out file')
    parser.add_argument('--endpoint', type=str, required=True, help='Endpoint for LLM')
    parser.add_argument('--token', type=str, required=False, default="ollama", help='Token for endpoint')
    parser.add_argument('--skip', action='store_true', help='Skip processing if flag is set')

    
    # Parse the arguments
    args = parser.parse_args()

    # Print the values of the arguments
    print(args)
    
    with open(args.in_file) as f:
        j = json.load(f)
        
    task_description = j['settings']['task_description']
    if task_description == 'SELECT':
        obj = AnswerSelectSchema
    elif task_description == 'EXPLAIN':
        obj = AnswerExplainSchema
    elif task_description == 'CONFIDENCE':
        obj = AnswerConfidenceSchema
    elif task_description == 'JUSTIFY':
        obj = AnswerJustifySchema
        
    path2 = args.out_file
    if os.path.exists(path2):
        with open(path2, 'r') as f:
            logs = json.load(f)
            examined_ids = set([res['query_id'] for res in logs['responses']])
            responses = logs['responses']
    else:
        examined_ids = set()
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        logs = {'settings': j['settings'], 'responses': []}
        responses = []    
        
    response_parser = PydanticOutputParser(pydantic_object=obj)        
    
    for no, prompt in enumerate(j['prompts']):
        # if no < 100:
        #     continue
        # print(no, prompt['query_id'], len(prompt['prompt']))
        if no % 10 == 0:
            print('Query {}/{}\r'.format(no, len(j['prompts'])), end='')
            logs['responses'] = responses
            with open(path2, 'w') as f:
                f.write(json.dumps(logs, indent=4))
                
        if prompt['query_id'] in examined_ids: #TODO: Change if Justifications
            continue
        # if prompt['reference_no'] != 0:
        #     continue

        if args.skip and len(prompt['options'])==1:
            query_time = time()
            answer = prompt['options'][0]
            explanation = None
            confidence = 1.0
            query_time = time() - query_time
        
        else:
            query_time = time()
            
            messages = [("system", "You are a helpful assistant trained in Entity Matching."),
                        ("human", prompt['prompt'])]
    
            try:
                llm = ChatOpenAI(model_name=args.model, openai_api_base=args.endpoint,
                                 openai_api_key=args.token, temperature=0.0, 
                                 timeout=120)
                response = llm.invoke(messages)
                
                result = response_parser.parse(response.content)
                if task_description != 'JUSTIFY':
                    answer = result.answer
                else:
                    answer = "[{}]".format(prompt['reference_no'])
                explanation = result.explanation
                confidence = getattr(result, 'confidence', 1.0)
                
            except httpx.TimeoutException:
                answer = None
                explanation = None
                confidence = None
                
            except Exception:
                # print(response.content)
                # traceback.print_exc() 
                answer = response.content
                explanation = None
                confidence = 1.0
                # exit(0)
                
            query_time = time() - query_time
        
        log = {'dataset': args.dataset, 'query_id': prompt['query_id'], 
               'ground_truth': prompt['ground_truth'], 
               'ground_answer': prompt['ground_answer'], 
               'answer': answer, 'explanation': explanation,
               'confidence': confidence, 'time': query_time
               }
        if 'reference' in prompt:
            log['reference'] = prompt['reference']
            log['reference_no'] = prompt['reference_no']
        responses.append(log)

    # print('\nMissed: ', missed)
    # j['settings']['model'] = args.model
    # path2 = args.out_file
    # os.makedirs(os.path.dirname(path2), exist_ok=True)
    with open(path2, 'w') as f:
        # logs = {'settings': j['settings'], 'responses': responses}
        logs['responses'] = responses
        f.write(json.dumps(logs, indent=4))