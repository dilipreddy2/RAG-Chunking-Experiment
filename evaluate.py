
import pandas as pd

# Manually assess whether the retrieved result
# is relevant to its question.
results = [
    ["What is RAG?", "Fixed-size", "Relevant"],
    ["What is RAG?", "Overlapping", "Partially relevant"],
    ["What is RAG?", "Structure-aware", "Not relevant"],

    ["How does RAG work?", "Fixed-size", "Partially relevant"],
    ["How does RAG work?", "Overlapping", "Relevant"],
    ["How does RAG work?", "Structure-aware", "Not relevant"],

    ["What is overlapping chunking?", "Fixed-size", "Relevant"],
    ["What is overlapping chunking?", "Overlapping", "Relevant"],
    ["What is overlapping chunking?", "Structure-aware", "Relevant"],

    ["Advantages of structure-aware?", "Fixed-size", "Relevant"],
    ["Advantages of structure-aware?", "Overlapping", "Relevant"],
    ["Advantages of structure-aware?", "Structure-aware", "Relevant"],

    ["Challenges of RAG?", "Fixed-size", "Not relevant"],
    ["Challenges of RAG?", "Overlapping", "Not relevant"],
    ["Challenges of RAG?", "Structure-aware", "Not relevant"]
]

df = pd.DataFrame(
    results,
    columns=["Question", "Strategy", "Relevance"]
)

# Summarize relevance counts
summary = pd.crosstab(
    df["Strategy"],
    df["Relevance"]
)

print("Retrieval Relevance Evaluation")
print("==============================")
print(summary)

# Save evaluation
df.to_csv("relevance_results.csv", index=False)
summary.to_csv("relevance_summary.csv")

print("\nEvaluation files saved successfully!")
