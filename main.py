import chromadb
from chromadb.utils import embedding_functions       # to change embedding model
chroma_client = chromadb.Client()

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)
collection = chroma_client.create_collection(name = "restaurant_menu", embedding_function=sentence_transformer_ef)

import pandas as pd
import numpy as np
import csv

with open('C:\\Users\\DELL\\Downloads\\menu_items.csv') as file:
    lines = csv.reader(file)

    documents = []
    metadetas = []
    ids = []
    id = 1

    for i, line in enumerate(lines):
        if i == 0:                      # Skipping header
            continue

        documents.append(line[1])
        metadetas.append({"item_id":line[0]})
        ids.append(str(id))
        id = id + 1

collection.add(
    documents=documents,
    metadatas=metadetas,
    ids=ids
)

results = collection.query(
    query_texts=('vermicleii'),
    n_results=10
)
print(results['documents'])

client = chromadb.PersistentClient(path="vectordb")
# creates folder vectordb with db in it

"""Now, we can try different language models to see which one gives better results
The crux - Embeddings are the key here

model_name="all-mpnet-base-v2" the best model"""
