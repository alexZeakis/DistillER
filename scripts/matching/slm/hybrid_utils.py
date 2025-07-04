import random, os, argparse, logging, sys, torch
import numpy as np
from enum import Enum
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                        RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
import json
import re

def initialize_gpu_seed(device:str, seed: int):
    device, n_gpu = setup_gpu(device)

    init_seed_everywhere(seed, n_gpu)

    return device, n_gpu


def init_seed_everywhere(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_gpu(device):
    # Setup GPU parameters
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.info("We use the device: '{}' and {} gpu's. Important: distributed and 16-bits training "
                 "is currently not implemented! ".format(device, n_gpu))

    return device, n_gpu

class DataType(Enum):
    TRAINING = "Training"
    EVALUATION = "Evaluation"
    TEST = "Test"


def load_data(examples, tokenizer, model_type):
    sep_token_extra=bool(model_type in ['roberta']),
    
    corpus = []
    corpus_ids = {}
    for (ex_index, example) in enumerate(examples):
        corpus_ids[(ex_index, 0)] = len(corpus)
        corpus += [tokenizer.tokenize(example.text_a)]
        corpus_ids[(ex_index, 1)] = len(corpus)
        corpus += [tokenizer.tokenize(example.text_b)]
        
        
    max_seq_length = min(get_max_sequence_length_from_dataset(examples, tokenizer),
                        tokenizer.model_max_length)
    special_tokens_count = 4 if sep_token_extra else 3
    max_tokens = max_seq_length - special_tokens_count


    features = {}
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        if example.id_a not in features:
            features[example.id_a] = {}
        if len(tokens_a) + len(tokens_b) <= max_tokens:
            features[example.id_a][example.id_b] = False
        else:
            features[example.id_a][example.id_b] = True
    return features
    
class InputExample(object):
    def __init__(self, guid, left_id, right_id, text_a, text_b, label, dataset):
        self.guid = guid
        self.id_a = left_id
        self.id_b = right_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.dataset = dataset


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, left_id=None, right_id=None, dataset=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.left_id = left_id
        self.right_id = right_id
        self.dataset = dataset


class DeepMatcherProcessor(object):
    """Processor for preprocessed DeepMatcher data sets (abt_buy, company, etc.)"""

    def get_train_examples(self, data_name):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_name, "train.csv")).fillna(""), 
            "train")

    def get_dev_examples(self, data_name):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_name, "valid.csv")).fillna(""),
            "dev")

    def get_test_examples(self, data_name):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_name, "test.csv")).fillna(""),
            "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        
        examples = []
        for index, row in df.iterrows():
            guid = "%s-%s" % (set_type, index)
            examples.append(
                InputExample(guid, row['Left_ID'], row['Right_ID'],
                             row['Left_Text'], row['Right_Text'], 
                             label=str(row['Label']),
                             dataset=int(row['Dataset'][-1])))
                
        return examples
    
            
def get_max_sequence_length_from_dataset(examples, tokenizer, sep_token_extra=False):
    lengths = []
    for example in examples:
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b) if example.text_b else None

        if tokens_b:
            special_tokens_count = 4 if sep_token_extra else 3
            total_length = len(tokens_a) + len(tokens_b) + special_tokens_count
        else:
            special_tokens_count = 3 if sep_token_extra else 2
            total_length = len(tokens_a) + special_tokens_count

        lengths.append(total_length)
    return max(lengths)






class Config():
    MODEL_CLASSES = {
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'sminilm': (None, AutoModelForSequenceClassification, AutoTokenizer),
    }




def read_arguments_train():
    parser = argparse.ArgumentParser(description='Run training with following arguments')

    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--log_dir', default=None, type=str, required=True)
    parser.add_argument('--data_name', default=None, type=str, required=True)
    parser.add_argument('--slm_predictions', default=None, type=str, required=True)
    parser.add_argument('--llm_predictions', default=None, type=str, required=True)
    parser.add_argument('--llm_prompts', default=None, type=str, required=True)
    parser.add_argument('--model_name_or_path', default="pre_trained_model/bert-base-uncased", type=str, required=True)
    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument('--device', default="cuda:1", required=False, type=str)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    args.data_path = os.path.join(args.data_dir, args.data_name)

    return args


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        datefmt="%H:%M:%S",
                        stream=sys.stdout,
                        #filename='log_file_name.log',
                        )

    logging.getLogger('bert-classifier-entity-matching')
    
def load_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return model, tokenizer






def get_prompts_for_json(path):
    j = json.load(open(path))
    # ground_results = {}
    options = {}
    for prompt in j['prompts']:
        options[prompt['query_id']] = prompt['options']
    return options


def get_scores_from_ft_json(path):
    j = json.load(open(path))
    # ground_results = {}
    predictions = {}
    for res in j['responses']:
        # print(res)

        answer = find_integer(res['response'])
        if len(answer) > 0:
            answer = answer['answer']
        
        # ground_results[res['query_id']] = res['answer']
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            qid = int(res['query_id'].split("_")[2])
            predictions[qid] = answer
            
    # return predictions, ground_results
    return predictions


def get_scores_from_json(path):
    j = json.load(open(path))
    
    # ground_results = {}
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
        # if res['ground_answer'] == -1: #TODO: Check for certainty
        #     ground_answer = 0
        
        # ground_results[res['query_id']] =  res['ground_answer']
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                answer = 0
            predictions[res['query_id']] = answer
            
    # return predictions, ground_results
    return predictions

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