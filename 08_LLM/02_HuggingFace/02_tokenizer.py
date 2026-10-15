from transformers import AutoTokenizer

#Load the model
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

#Input text
text = "Robotics is my field"

#Gnerate tokens
tokens = tokenizer.tokenize(text)

print("Tokens")
print(tokens)

#Generate token ids
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("token ids")
print(token_ids)

# Tokenize and convert to model input format
inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("\nModel inputs:")
print(inputs)


print("\nInput IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])