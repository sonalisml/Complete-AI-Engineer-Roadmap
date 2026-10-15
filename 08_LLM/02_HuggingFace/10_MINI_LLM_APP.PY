#BUIlding a mini text generation application

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#1. Load pretrained model
model_name ="distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# 2. Define a reusable generation function
def generate_text(prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.inference_mode():
         output_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature = 0.7,
            top_p = 0.9 ,
            do_sample = True,
            pad_token_id = tokenizer.eos_token_id
         )
    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens = True
    )

# 3. Test the application
prompt = "A robot can help humans by"
result = generate_text(prompt, max_new_tokens=40)
print("Prompt:", prompt)
print("Generated text:", result)