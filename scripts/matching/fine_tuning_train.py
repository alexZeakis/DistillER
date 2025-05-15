import os 
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
os.environ["HF_HOME"] = "/mnt/data/ollama_custom/"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/ollama_custom/"

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import argparse
import json
import torch



def fine_tune(model_name, input_file, out_dir, out_name, log_file):
    models = {'llama3.1': "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
              'llama3.1-instruct': "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
              'mistral-v0.3': "unsloth/mistral-7b-v0.3-bnb-4bit",
              'mistral-v0.3-instruct': 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
             }
    model = models[model_name]
    
    instruct = 'instruct' in model_name
    
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = not instruct # Use 4bit quantization to reduce memory usage. Can be False.
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    if instruct:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
                          "up_proj", "down_proj", "embed_tokens", "lm_head"]  # Added embed_tokens and lm_head when not Instruct Model        
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = target_modules,
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    ## ADDED WHEN NOT INSTRUCT ##
    if not instruct:
        for param in model.model.model.embed_tokens.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True    
    ## ADDED WHEN NOT INSTRUCT ##
    
    if 'llama3' in model_name:
        chat_template = 'llama-3'
    elif 'mistral-v0.3' in model_name:
        chat_template = "mistral"
        
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = chat_template, # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False,
                                               add_generation_prompt = False)
                 for convo in convos]
        return { "text" : texts, }
    
    dataset = load_dataset("json",
                           data_files=input_file,
                           split="train") 
    
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    
    # Step 2: Tokenize the 'text' to produce 'input_ids'
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 3,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = out_dir+"_outputs",
        ),
    )
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    # Store initial GPU and memory stats in a dictionary
    memory_stats = {
        "GPU": gpu_stats.name,
        "Max memory (GB)": max_memory,
        "Initial reserved memory (GB)": start_gpu_memory,
    }
    
    trainer_stats = trainer.train()
    
    # Calculate memory usage and training stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    model.save_pretrained(out_dir)
    # model = model.to('cpu')
    # torch.cuda.empty_cache()
    # model.save_pretrained_gguf(out_name, tokenizer)
    # , quantization_method = "q4_k_m")
    
    # Update the dictionary with final stats
    memory_stats.update({
        "Training time (seconds)": trainer_stats.metrics['train_runtime'],
        "Training time (minutes)": round(trainer_stats.metrics['train_runtime'] / 60, 2),
        "Peak reserved memory (GB)": used_memory,
        "Memory used for training (GB)": used_memory_for_lora,
        "Memory usage % of max": used_percentage,
        "Training memory usage % of max": lora_percentage
    })
    
    print(memory_stats)
    with open(log_file, 'w') as f:
        f.write(json.dumps(memory_stats, indent=4))
    
    
    
if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--input_file', type=str, required=True, help='Input file')
    parser.add_argument('--log_file', type=str, required=True, help='Log directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--out_name', type=str, required=True, help='Output Model Name')
    parser.add_argument('--model', type=str, required=True, choices=['llama3.1', 'mistral-v0.3', 'llama3.1-instruct', "mistral-v0.3-instruct"], 
                        help='Name of output file.')
    
    args = parser.parse_args()
    fine_tune(args.model, args.input_file, args.out_dir, args.out_name, args.log_file)