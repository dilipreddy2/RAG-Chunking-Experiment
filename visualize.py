import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("relevance_results.csv")

summary = pd.crosstab(
    df["Strategy"],
    df["Relevance"]
)

summary = summary[
    ["Relevant", "Partially relevant", "Not relevant"]
]

summary.plot(
    kind="bar",
    stacked=True,
    title="Retrieval Relevance by Chunking Strategy",
    ylabel="Number of Questions",
    xlabel="Chunking Strategy"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("relevance_comparison.png", dpi=300)
plt.show()

print("Relevance chart saved successfully!")