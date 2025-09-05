import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from trl import GRPOConfig, GRPOTrainer
import random
import re

# def random_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that assigns random scores for testing purposes."""
#     for c in completions:
#         print(c)
#     return [random.uniform(0, 1) for _ in completions]

def extract_digit_response(completion):
    # First try to find patterns like [0], [1], ...
     match = re.search(r"\[(\d+)\]", completion)
     if match:
         return int(match.group(1))
     
     # If none found, fallback to standalone digits
     match = re.search(r"\b(\d+)\b", completion)
     if match:
         return int(match.group(1))
     
     # Nothing found
     return None



def digit_count_reward_single(completion):
    digits = re.findall(r'\[\d\]', completion)
    n = len(digits)
    
    if n == 0:
        return 0.0
    else:
        return 1.0 / n
    
def digit_count_reward(completions, **kwargs) -> list[float]:
    return [digit_count_reward_single(c) for c in completions]

def len_reward_single(completion):
    stripped = completion.strip()  # remove leading/trailing whitespace/newlines
    n = len(stripped)
    
    if n == 0:
        return 0.0
    elif n in [1,2]:
        return 1.0
    else:
        return 3.0 / n
    
def len_reward(completions, **kwargs) -> list[float]:
    return [len_reward_single(c) for c in completions]

def correct_reward_single(completion, true):
    pred = extract_digit_response(completion)
    return 1.0 * (pred == true)

# def total_reward(completions, **kwargs) -> list[float]:
def total_reward(prompts, completions, true, **kwargs) -> list[float]:
    func = [digit_count_reward_single, len_reward_single, correct_reward_single]
    weights = [.25, .25, .50]
    
    scores = []
    # w = 1.0 / len(func)
    for c, p, t in zip(completions, prompts, true):
        score = 0
        for w, f in zip(weights, func):
            if f in [correct_reward_single]: # 2-argument function
                score += w * f(c, t)
            else: # 1-argument function
                score += w * f(c)
        scores.append(score)
        
    return scores        

def train_grpo(model_name, input_file, out_dir, log_file, eval_split_ratio=0.0):
    models = {
        'llama3.1': "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        'llama3.1-instruct': "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        'mistral-v0.3': "unsloth/mistral-7b-v0.3-bnb-4bit",
        'mistral-v0.3-instruct': 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
    }
    pretrained = model_name in models
    
    if pretrained:
        model_name_true = models[model_name]
    else:
        model_name_true = model_name

    max_seq_length = 1024
    instruct = "instruct" in model_name_true.lower()

    # 1️⃣ Load pretrained model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_true,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if pretrained: # SFT models have already LoRA
        # 2️⃣ Configure LoRA target modules
        if instruct:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",
                              "embed_tokens", "lm_head"]
    
        # 3️⃣ Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=target_modules,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            max_seq_length=max_seq_length,
        )

    # 4️⃣ For non-instruct models, make embedding & lm_head trainable
    if not instruct:
        for param in model.model.model.embed_tokens.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # 5️⃣ Load dataset from JSON file
    dataset = load_dataset("json", data_files=input_file, split="train")
    data_list = dataset.to_list()

    if eval_split_ratio > 0.0:
        train_samples, eval_samples = train_test_split(
            data_list, test_size=eval_split_ratio, random_state=42
        )
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_samples),
            "eval": Dataset.from_list(eval_samples),
        })
    else:
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(data_list),
        })

    # 6️⃣ GRPO Config
    # max_prompt_length = 1024
    # max_seq_length = 1024
    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        # optim = "paged_adamw_8bit",
        optim="adamw_8bit",
        logging_steps = 10,
        
        # per_device_train_batch_size = 2,
        # gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        # num_generations = 1, # Decrease if out of memory
        # # max_prompt_length = max_prompt_length,
        # # max_completion_length = 2048 - max_prompt_length,
        # num_train_epochs = 3, # Set to 1 for a full training run
        # # max_steps = 250,
        # # save_steps = 250,
        # # save_strategy="epoch",
        
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        # num_generations=4,
        num_train_epochs = 1, #TODO: Change to 3
        # num_generations = 1,
        # max_steps = 2,   # to match ~3 epochs of DPO
        
        max_grad_norm = 0.1,
        report_to = "none", # Can use Weights & Biases
        output_dir = "outputs",
    )    
    
    # 7️⃣ GRPO Trainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            total_reward,
        ],
        args = training_args,
        train_dataset = dataset,
    )

    # 7️⃣ GPU memory stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    memory_stats = {
        "GPU": gpu_stats.name,
        "Max memory (GB)": max_memory,
        "Initial reserved memory (GB)": start_gpu_memory,
    }

    # 8️⃣ Train
    trainer_stats = trainer.train()

    # 9️⃣ GPU memory usage after training
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_training = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    training_percentage = round(used_memory_for_training / max_memory * 100, 3)

    memory_stats.update({
        "Training time (seconds)": trainer_stats.metrics['train_runtime'],
        "Training time (minutes)": round(trainer_stats.metrics['train_runtime'] / 60, 2),
        "Peak reserved memory (GB)": used_memory,
        "Memory used for training (GB)": used_memory_for_training,
        "Memory usage % of max": used_percentage,
        "Training memory usage % of max": training_percentage
    })

    # 10️⃣ Save model & tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"✅ Model saved to {out_dir}")
    print(memory_stats)
    with open(log_file, 'w') as f:
        f.write(json.dumps(memory_stats, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO fine-tuning script")
    parser.add_argument("--model", type=str, required=True,
                        help="Model key, e.g., 'llama3.1-instruct'")
    parser.add_argument("--input_file", type=str, required=True, help="JSON input dataset")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_file", type=str, required=True, help="Log JSON file path")
    parser.add_argument("--eval_split_ratio", type=float, default=0.0,
                        help="Fraction of data to hold out for evaluation (0.0 means all for training)")

    args = parser.parse_args()
    train_grpo(args.model, args.input_file, args.out_dir, args.log_file, args.eval_split_ratio)
