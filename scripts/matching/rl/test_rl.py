import os
import json
import argparse
import torch
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_single_response(model, tokenizer, prompt, max_new_tokens=64, device="cuda"):
    """
    Generate a response for a single prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # if response == "":
    #   print(generated_tokens)
    # print(response)
    return response

def test_model(model_path, input_file, out_file, dataset, max_new_tokens=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    
    with open(input_file, "r") as f:
        data = json.load(f)

    # all_prompts = [item["conversations"] for item in data]
    all_prompts = [item["prompt"] for item in data]
    output = []

    for idx, prompt_set in enumerate(all_prompts):
        if idx % 50 == 0:
            print(f"Processing {idx}/{len(all_prompts)}\r", end="")
        # if len(output) >= 20:
        #     break
            
            
        qtime = time()
        response = generate_single_response(model, tokenizer, prompt_set, max_new_tokens, device)
        qtime = time() - qtime

        output.append({
            "query_id": data[idx]["id"],
            "ground_truth": data[idx].get("gt", ""),
            "answer": data[idx].get("true", ""),
            "response": response,
            "time": qtime
        })

    # Save results
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        logs = {"settings": {"model": model_path, 'dataset': dataset}, "responses": output}
        json.dump(logs, f, indent=4)
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a fine-tuned language model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument("--input_file", type=str, required=True, help="JSON input file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to save output")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate per prompt")
    args = parser.parse_args()

    test_model(args.model_path, args.input_file, args.out_file, args.dataset, args.max_new_tokens)
