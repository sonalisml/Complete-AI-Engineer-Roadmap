from transformers import AutoTokenizer, AutoModel

#Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
     "distilbert-base-uncased"
)

#Load pretrained model
model = AutoModel.from_pretrained(
     "distilbert-base-uncased"
)

print("Tokenizer loaded")
print("Model loaded")

print(type(tokenizer))
print(type(model))