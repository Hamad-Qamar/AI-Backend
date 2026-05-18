from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset
products = pd.read_csv("products.csv")

# Combine text fields
products["search_text"] = (
    products["item_name"].fillna("") + " " +
    products["category"].fillna("") + " " +
    products["brand"].fillna("") + " " +
    products["quality_grade"].fillna("") + " " +
    products["notes"].fillna("")
)

# Convert to embeddings
embeddings = model.encode(products["search_text"].tolist())

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "products.index")

print("Index Created Successfully")