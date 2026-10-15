from sentence_transformers import SentenceTransformer

sentences = [
    "How can I improve model accuracy?",
    "Methods for improving neural network performance."
]

model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

for model_name in model_names:
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        sentences,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    print("\nModel:", model_name)
    print("Embedding shape:", embeddings.shape)
    print("First vector's first 5 values:", embeddings[0, :5])