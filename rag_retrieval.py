import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
with open("data/sample_document.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define chunking strategies
fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

overlap_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

fixed_chunks = fixed_splitter.split_text(text)
overlap_chunks = overlap_splitter.split_text(text)
structure_chunks = [
    section.strip()
    for section in text.split("\n## ")
    if section.strip()
]

strategies = {
    "Fixed-size": fixed_chunks,
    "Overlapping": overlap_chunks,
    "Structure-aware": structure_chunks
}

# Test questions
questions = [
    "What is RAG?",
    "How does RAG work?",
    "What is overlapping chunking?",
    "What are the advantages of structure-aware chunking?",
    "What are the challenges of RAG?"
]

# Perform retrieval
for strategy, chunks in strategies.items():
    print("\n" + "=" * 60)
    print("STRATEGY:", strategy)
    print("Number of chunks:", len(chunks))
    print("=" * 60)

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    for question in questions:
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = index.search(query_embedding, 1)

        print("\nQuestion:", question)
        print("Distance:", round(float(distances[0][0]), 4))
        print("Retrieved text:")
        print(chunks[indices[0][0]])
        print("-" * 50)