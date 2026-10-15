import time 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

texts = [
    "Robotics is useful",
    "Artificial intelligence is transforming healthcare",
    "Deep learning helps computers learn patterns"
]

# Tokenize all inputs as a batch
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

start = time.perf_counter()

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

elapsed = time.perf_counter() - start

for i, output in enumerate(output_ids):
    print(f"\nInput {i + 1}: {texts[i]}")
    print("Output:", tokenizer.decode(
        output,
        skip_special_tokens=True
    ))

print(f"\nBatch generation time: {elapsed:.2f} seconds")