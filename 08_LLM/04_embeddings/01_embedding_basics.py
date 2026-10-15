from sentence_transformers import SentenceTransformers

model = SentenceTransformers("sentence_transformers/all-MiniLM-L6-v2")

sentences = [
    "I want to learn deep learning",
    "i want to play",
    "I want to study"
]

embeddings = model.encode(sentences)

print(embeddings.shape)

for sentences, embeddings in zip(sentences, embeddings):
    print(sentence)
    print(embeddings[:5])