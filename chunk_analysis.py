import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
with open("data/sample_document.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test questions
questions = [
    "What is RAG?",
    "How does RAG work?",
    "What is overlapping chunking?",
    "What are the advantages of structure-aware chunking?",
    "What are the challenges of RAG?"
]

results = []

# Compare three chunk sizes
for size in [300, 500, 800]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=0
    )

    chunks = splitter.split_text(text)

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    distances_list = []

    for question in questions:
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = index.search(query_embedding, 1)
        distances_list.append(float(distances[0][0]))

    results.append({
        "Chunk Size": size,
        "Number of Chunks": len(chunks),
        "Average Chunk Length": round(
            sum(len(c) for c in chunks) / len(chunks), 2
        ),
        "Average Retrieval Distance": round(
            sum(distances_list) / len(distances_list), 4
        )
    })

df = pd.DataFrame(results)

print("\nChunk Size Comparison")
print("=====================")
print(df.to_string(index=False))

df.to_csv("chunk_size_results.csv", index=False)

print("\nResults saved to chunk_size_results.csv")