#==============
#Import libraries
#==============
import torch
from transformers import AutoTokenizer, AutoModel
#==============

#Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#Load model
model = AutoModel.from_pretrained("bert-base-uncased")

text = "Robotics is my field"

#Convert token ids to token ids
inputs = tokenizer(text, return_tesnors ="pt")
print("InputIDs")
print(inputs["input_ids"])
print(inputs["input_ids"].shape)


# Pass tokens through BERT
with torch.no_grad():
    outputs = model(**inputs)
# Get embeddings
embeddings = outputs.last_hidden_state

print("\nEmbedding shape:")
print(embeddings.shape)