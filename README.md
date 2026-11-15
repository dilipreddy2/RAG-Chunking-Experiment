# Experiment With Document Chunking Strategies in RAG

## 1. Introduction

This project explores different document chunking strategies used in Retrieval-Augmented Generation (RAG).

The main goal is to compare how fixed-size, overlapping, and structure-aware chunking affect document retrieval.

## 2. Chunking Strategies

- Fixed-size chunking: Splits text into chunks with a target size of 500 characters and no overlap.
- Overlapping chunking: Splits text into chunks with a target size of 500 characters and 100 characters of overlap.
- Structure-aware chunking: Splits the document based on section headings.

## 3. Tools and Technologies

- Python
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- Pandas
- Matplotlib

## 4. Experiments

The project compares:
- Number of chunks generated
- Embedding-based retrieval
- Retrieval relevance
- Different chunk sizes (300, 500, and 800 characters)

## 5. Results

The experiments show that chunking strategy affects retrieval results.

The relevance evaluation found:
- Fixed-size: 3 relevant, 1 partially relevant, 1 not relevant.
- Overlapping: 3 relevant, 1 partially relevant, 1 not relevant.
- Structure-aware: 2 relevant, 0 partially relevant, 3 not relevant.

The chunk-size experiment found that 300-character chunks had the lowest average FAISS L2 distance among the tested sizes.

## 6. Conclusion

Chunk size and chunking strategy influence document retrieval. Smaller chunks can provide more focused information, while overlapping chunks help preserve context between adjacent sections.

The best strategy depends on the document structure and retrieval requirements.