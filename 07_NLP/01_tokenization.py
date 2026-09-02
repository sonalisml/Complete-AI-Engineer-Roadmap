#========
#Tokenization
#========
from transformers import AutoTokenizer

#load a pretarined tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
text ="Robotics is my field"

tokens = tokenizer.tokenize(text)

print("origial text")
print(text)

print("tokens are :")
print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("token ids are:")
print(token_ids)