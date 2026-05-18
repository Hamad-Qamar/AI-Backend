import pytest

pytest.skip(
    "Legacy FAISS index smoke script (products.csv) — not a pytest test. "
    "Phase-2 uses materials_master/pricing_data and a different index path.",
    allow_module_level=True,
)

from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load data
products = pd.read_csv("products.csv")

# load index
index = faiss.read_index("products.index")

# test query
query = "cement for foundation in lahore"

query_embedding = model.encode([query])

distances, indices = index.search(np.array(query_embedding), 5)

print("Top Recommendations:\n")

for i in indices[0]:
    print(products.iloc[i]["title"])