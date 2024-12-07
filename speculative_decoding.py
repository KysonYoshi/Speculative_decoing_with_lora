import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import torch.nn.functional as F

import json
import time
from tqdm import tqdm

def load_models():
    # Load the draft model (small LoRA fine-tuned model)
    draft_model_path = "./qwen2.5_0.5B_lora_gsm8k"
    base_draft_model_path = "./QwenQwen2.5-0.5B-Instruct"
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_path, trust_remote_code=True)
    base_draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    draft_model = PeftModel.from_pretrained(base_draft_model, draft_model_path)

    # Set pad_token_id if not set
    if draft_tokenizer.pad_token is None:
        draft_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_draft_model.config.pad_token_id = draft_tokenizer.pad_token_id

    # Load the target model (large 7B model)
    target_model_path = "/home/cl6933/layerskipLora-main/QwenQwen2.5-7B-Instruct"
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Set pad_token_id if not set
    if target_tokenizer.pad_token is None:
        target_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    target_model.config.pad_token_id = target_tokenizer.pad_token_id

    return draft_model, draft_tokenizer, target_model, target_tokenizer

def speculative_decoding(
    prompt, 
    draft_model, 
    draft_tokenizer, 
    target_model, 
    target_tokenizer, 
    max_length=128, 
    gamma=4, 
    temp=1.0
):
    """
    Speculative Decoding:
    - `gamma`: Number of draft tokens to generate per step.
    - `prompt`: Initial input text.
    - `draft_model` and `target_model`: The draft and target models for decoding.
    """
    # Initialization
    draft_device = draft_model.device
    target_device = target_model.device
    inputs = draft_tokenizer(prompt, return_tensors="pt")
    prefix = inputs.input_ids.to(draft_device)
    attention_mask = inputs.attention_mask.to(draft_device)
    max_tokens = max_length + prefix.shape[1]
    total_draft_tokens = 0
    accepted_draft_tokens = 0
    
    while prefix.shape[1] < max_tokens:
        prefix_len = prefix.shape[1]
        
        # Step 1: Draft model generates `gamma` tokens
        draft_outputs = draft_model.generate(
            prefix, 
            max_new_tokens=gamma,
            do_sample=False,
            attention_mask=attention_mask,  # Add attention_mask
            pad_token_id=draft_tokenizer.pad_token_id  # Explicitly set pad_token_id
        )
        # Extract only the newly generated draft tokens
        draft_tokens = draft_outputs[:, prefix_len:]

        # Step 2: Target model evaluates probabilities for the same sequence
        extended_inputs = torch.cat([prefix, draft_tokens], dim=1).to(target_device)
        extended_attention_mask = torch.cat([
            attention_mask, 
            torch.ones(draft_tokens.shape, dtype=torch.long).to(draft_device)
        ], dim=1).to(target_device)
        
        target_outputs = target_model(extended_inputs, attention_mask=extended_attention_mask)
        target_logits = target_outputs.logits[:, prefix_len-1:, :]  # Only new tokens
        target_tokens = torch.argmax(target_logits, dim=-1)
        
        # Step 3: Verification process
        n = gamma if len(draft_tokens[0]) == gamma else len(draft_tokens[0])
        for i in range(len(draft_tokens[0])):
            # Get the draft token at position `i` and calculate probabilities
            draft_token = draft_tokens[:, i]  # Token proposed by draft model
            target_token = target_tokens[:, i]

            if draft_token != target_token:
                n = i + 1
                break

        # Update statistics
        total_draft_tokens += gamma  # Increment total tokens generated
        accepted_draft_tokens += n  # Increment accepted tokens
        
        # Accept tokens up to position `n`
        prefix = torch.cat([prefix, target_tokens[:, :n]], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, n), dtype=torch.long).to(draft_device)], dim=1)
        
        if prefix.shape[1] >= max_tokens:
            break
        
        if target_tokenizer.eos_token_id in prefix.tolist():
            break

    draft_pass_rate = accepted_draft_tokens / total_draft_tokens * 100 if total_draft_tokens > 0 else 0
    print(f"passed rate: {draft_pass_rate}%")
    print(total_draft_tokens)
    # Decode final result
    decoded_text = draft_tokenizer.decode(prefix.squeeze(0), skip_special_tokens=True)
    return decoded_text, draft_pass_rate

def main():
    # Model setup
    # Load models and tokenizers
    draft_model, draft_tokenizer, target_model, target_tokenizer = load_models()

    # Input and output files
    input_file = "data/gsm8k.jsonl"  # Input dataset file
    total_rate = 0

    # Load all questions into memory
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = [json.loads(line.strip()) for line in infile]
    
    start_time = time.time()
    for i in tqdm(range(0, 10), desc="Processing Batches"):
        # Prepare batch
        questions = data[i]["question"]

        # Perform speculative decoding
        generated_text, draft_pass_rate = speculative_decoding(
            questions, draft_model, draft_tokenizer, target_model, target_tokenizer, max_length=128
        )            

        print(generated_text)
        total_rate += draft_pass_rate
    
    print(f'rate: {total_rate/10}') 
    end_time = time.time()  
    execution_time = end_time - start_time
    print(f'time: {execution_time}')

if __name__ == '__main__':
    main()
