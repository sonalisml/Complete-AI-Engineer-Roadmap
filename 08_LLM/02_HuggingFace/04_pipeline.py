#Hugging Face Pipeline
#Now we'll learn the simplest way to use a pretrained model with Hugging Face.
from transformers import pipeline

#Create a sentiment analysisi pipeline
classifier = pipeline("sentiment-analysis")

#Input text
text = "I really enjoy learning ai and deep learning"

result = classifier(text)

print(result)
