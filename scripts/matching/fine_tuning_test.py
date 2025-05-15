import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["HF_HOME"] = "/mnt/data/entity_matching_embeddings/positionBiasER/models/"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/entity_matching_embeddings/positionBiasER/models/"

import os
import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import argparse
from time import time



def generate_single_response(model, tokenizer, message):
    """
    Generate a response for a single message set.
    
    Args:
    - model: Loaded model for inference.
    - tokenizer: Loaded tokenizer.
    - message: A list of dictionaries representing a single message set.

    Returns:
    - generated_text: The generated response as a string.
    """
    # Tokenize the message
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    input_ids = inputs.to("cuda")
    attention_mask = (input_ids != tokenizer.pad_token_id).to("cuda")

    # Generate response
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
        use_cache=True
    )

    # Decode the generated response
    response = outputs[0][input_ids.shape[-1]:]  # Trim input tokens
    generated_text = tokenizer.decode(response, skip_special_tokens=True)
    
    return generated_text

def test_model(model_path, input_file, out_file, dataset):
    max_seq_length = 2048
    dtype = None  # Auto-detection
    # load_in_4bit = True
    instruct = 'instruct' in model_path
    load_in_4bit = not instruct # Use 4bit quantization to reduce memory usage. Can be False.

    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    FastLanguageModel.for_inference(model)

    if 'llama3' in model_path:
        chat_template = 'llama-3'
    elif 'mistral-v0.3' in model_path:
        chat_template = "mistral"

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = chat_template, # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
    )    
    
    with open(input_file) as f:
        lines = json.load(f)
        
    # Process each message set one by one
    all_messages = [j['conversations'] for j in lines]
    # responses = []
    
    output = []
    for no, message in enumerate(all_messages):
        if no % 50 == 0:
            print('Query {}/{}\r'.format(no, len(all_messages)), end='')
        qtime = time()
        response = generate_single_response(model, tokenizer, message)
        #qtime = qtime - time()
        qtime = time() - qtime
        out = {'query_id': lines[no]['id'],
               'ground_truth': lines[no]['gt'],
               'answer': lines[no]['true'],
               'response': response,
               'time': qtime
            }
        output.append(out)
    print(f"\tMessages: {len(all_messages)}, Responses: {len(output)}")
    
    # Save the responses to the specified log directory
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        logs = {'settings': {'model': model_path, 'dataset': dataset}, 'responses': output}
        f.write(json.dumps(logs, indent=4))

if __name__ == '__main__':
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_file', type=str, required=True, help='Output File')
    parser.add_argument('--input_file', type=str, required=True, help='Input File')
    
    
    #path = '/mnt/data/entity_matching_embeddings/positionBiasER/models/llama_3_1_7b_lora'
    # Parse the arguments
    args = parser.parse_args()

    test_model(args.model_path, args.input_file, args.out_file, args.dataset)