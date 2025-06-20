import os
import pandas as pd
import argparse
import json
from utils import settings
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from structured_responses import AnswerSelectSchema, AnswerExplainSchema,AnswerConfidenceSchema
import traceback
import re
    
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

def prepare_description(df, query=False, serialization='DITTO'):
    desc = ''
    for no, (index, row) in enumerate(df.iterrows()):
        if not query:
            desc += f'[{no+1}] '
        desc += serialize(row, serialization) + '\n'
    return desc

def prepare_select_prompt(df1, df2, serialization='DITTO', task_description='SELECT'):
    
    entity_description = prepare_description(df1, True, serialization=serialization)
    entity_description = re.sub(r"[{}]", "", entity_description) 
    candidate_description = prepare_description(df2, serialization=serialization)
    candidate_description = re.sub(r"[{}]", "", candidate_description) 
    
    prompt = """Select a record from the following candidates that refers """ \
               """to the same real-world entity as the given record. """ \
               """\nGiven entity record: {} \nCandidate records:\n {}\n"""
    prompt = prompt.format(entity_description, candidate_description)
    
    if task_description == 'SELECT':
        obj = AnswerSelectSchema
    elif task_description == 'EXPLAIN':
        obj = AnswerExplainSchema
    elif task_description == 'CONFIDENCE':
        obj = AnswerConfidenceSchema
        
    parser = PydanticOutputParser(pydantic_object=obj)
    prompt += "Please respond in the correct format:\n{format_instructions}"
    prompt = ChatPromptTemplate.from_messages([("human", prompt)])
    formatted_prompt = prompt.format_messages(format_instructions=parser.get_format_instructions())
    human_prompt = formatted_prompt[0].content

    return human_prompt

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_file', type=str, required=True, help='Out file')
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--sample_file', type=str, required=True, help='Sample file')
    
    parser.add_argument('--seed', type=int, default=1924, required=False, help='Seed for shuffling')
    parser.add_argument('--serialization', type=str, choices=['DITTO', 'schema_agnostic', 'llm'],
                        required=True, help='Serialization Strategy')
    parser.add_argument('--task_description', type=str, choices=['SELECT', 'EXPLAIN', 'CONFIDENCE'],
                        required=True, help='Task Description Strategy')
    parser.add_argument('--skip', action='store_true', help='Skip processing if flag is set')
    
    # Parse the arguments
    args = parser.parse_args()
    
    dataset = args.dataset
    serialization=args.serialization
    task_description=args.task_description
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
    
    #TODO: Recall to remove -1 labels!!!!
    logs = []
    lens = []
    
    for no, (key, cands) in enumerate(total_cands.items()):
        if no % 50 == 0:
            print('Query {}/{}\r'.format(no, len(total_cands)), end='')
            
        if args.skip and len(cands)==1: #do not add trivial prompts
            continue
        
        temp_df1 = pd.DataFrame(df1.loc[key]).T
        temp_df2 = df2.loc[cands]
        temp_df2 = shuffle_on_seed(temp_df2, seed)
        options = temp_df2.index
        true_answer = int(ground_truth.get(key, -1))
        answer = find_answer_on_shuffle(temp_df2, true_answer)
            
        prompt = prepare_select_prompt(temp_df1, temp_df2,
                                       serialization=serialization,
                                       task_description=task_description)
        
        log = {'query_id': key, 'ground_truth': true_answer,
               'ground_answer': answer, 'options': list(options), 'prompt': prompt,
               }
        logs.append(log)
        lens.append(len(prompt))
            
    settings = vars(args)
    if len(lens) != 0:
        lens = pd.Series(lens)
        settings['len_q99'] = lens.quantile(0.99)
        settings['len_q50'] = lens.quantile(0.50)
    
    path2 = args.out_file
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    with open(path2, 'w') as f:
        logs = {'settings': settings, 'prompts': logs}
        f.write(json.dumps(logs, indent=4))