import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("chunk_size_results.csv")

df.plot(
    x="Chunk Size",
    y="Average Retrieval Distance",
    kind="bar",
    title="Effect of Chunk Size on Retrieval Distance",
    legend=False,
    ylabel="Average FAISS L2 Distance"
)

plt.tight_layout()
plt.savefig("chunk_size_comparison.png", dpi=300)
plt.show()

print("Chunk size graph saved!")