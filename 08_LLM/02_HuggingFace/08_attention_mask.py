#Attention masks, padding, and batching
from transformers import AutoTokenizer, AutoCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

model.eval()

text = [
    "Robotics",
    "Artificial Intelligence"
]

#Tokenize both inputs together
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(
        text,
        padding = True,
        truncation = True,
        return_tensors = "pt"
)

print("inputs")
print(inputs["input_ids"])

print(inputs["attention_mask"])

#Run inference
with torch.no_grad(
    outputs = model(**inputs)
)

print("\nLogits shape:")
print(outputs.logits.shape)