
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "I want to learn deep learning.",
    "I want to study neural networks.",
    "The weather is sunny today."
]

# Convert sentences into embedding vectors
embeddings = model.encode(
    sentences,
    convert_to_tensor=True
)

# Compare the first sentence with every sentence
similarity_scores = cos_sim(
    embeddings[0],
    embeddings
)

print("Similarity scores:")

for sentence, score in zip(
    sentences,
    similarity_scores[0]
):
    print(f"{score.item():.4f} - {sentence}")
