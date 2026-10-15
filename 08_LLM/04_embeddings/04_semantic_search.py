from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 1. Load a pretrained embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Create a small knowledge base
documents = [
    "CNN models can classify lung CT images.",
    "Data augmentation can improve deep learning generalization.",
    "Cosine similarity compares the directions of two vectors.",
    "Transformers use self-attention to process token relationships.",
    "Precision and recall are important classification metrics."
]
# 3. Convert all documents into embeddings
document_embeddings = model.encode(
    documents,
    convert_to_tensor=True,
    normalize_embeddings=True
)
#ASk a question
query = "how can i reduce overfitting in a neural network ?"

#convert the question to an embedding
query_embedding = model.encode(
    query,
    convert_to_tensor = True,
    normalize_embeddings = True
)
#compare the query with every document
scores = cos_sim(query_embedding, documents_embeddings)[0]

#rank documents by similarity
ranked_indices = scores.argsort(decsending=True)

top_k =3
print(query)
print("Match documents")

for rank, index in enumerate(ranked_indices[:top_k], start=1):
    i = index.item()
    print(f"\nRank: {rank}")
    print(f"Similarity: {scores[i].item():.4f}")
    print(f"Document: {documents[i]}")