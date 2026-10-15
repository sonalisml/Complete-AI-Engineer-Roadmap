
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Create a small knowledge base
documents = [
    "CNN models classify lung CT images.",
    "Data augmentation can reduce overfitting.",
    "Transformers use self-attention.",
    "Precision and recall evaluate classifiers.",
    "Cosine similarity compares vector directions."
]

# 3. Generate Normalized Document embeddings
embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

# 4. FAISS expects a contiguous float32 NumPy array
embeddings = np.ascontiguousarray(
    embeddings, 
    dtype=np.float32
)

# 5. Create an exact inner-product index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# 6. Add document vectors to the index
index.add(embeddings)

# 7. Encode the user's question
query = "How can I reduce overfitting?"
query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)
query_embedding = np.ascontiguousarray(
    query_embedding, 
    dtype=np.float32
)

# 7. Retrieve the top 3 documents
scores, indices = index.search(query_embedding, k=3)

print("Question:", query)
print("\nTop matching documents:")

for score, idx in zip(scores[0], indices[0]):
    if idx != -1:
        print(f"{score:.4f} - {documents[idx]}")
