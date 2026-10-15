#-Let’s continue the Hugging Face → LLM path. Load a real Causal LLM- AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# 2. Load pretrained causal language model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

text = "Robotics is "

# 3. Input text
inputs = tokenizer(text, return_tensors = "pt") 
print("Input IDs:")
print(inputs["input_ids"])

#5.Generate new tokens
with torch.no_grad():
    output_ids = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_new_tokens=30
    )

# 6. Convert token IDs back to text
generated_text = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True
)

print("Generated text")
print(generated_text)