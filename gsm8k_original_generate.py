import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    # Model setup
    model_name_or_path = '/home/asperger/DPSD/models/Qwen2.5-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16
    )

    # Input and output files
    input_file = "data/gsm8k.jsonl"  # Input dataset file
    output_file = "data/gsm8k_generated.jsonl"  # Output file for storing results
    batch_size = 24  # Set batch size (adjust based on your GPU memory)

    # Load all questions into memory
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = [json.loads(line.strip()) for line in infile]

    # Prepare output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Add progress bar with tqdm
        for i in tqdm(range(0, len(data), batch_size), desc="Processing Batches"):
            # Prepare batch
            batch = data[i:i + batch_size]
            questions = [item["question"] for item in batch]

            # Tokenize batch
            inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate answers for the batch
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            # Decode and save answers
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for item, generated_text in zip(batch, generated_texts):
                item["generated_answer"] = generated_text
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()
