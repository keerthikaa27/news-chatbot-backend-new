import chromadb
import requests
from chromadb.config import Settings

#Config
JINA_API_KEY = "jina_c3559ea7e97844b884040a4822311e73UevqBdGSHekCci3aFAIsno5SPDjU"
COLLECTION_NAME = "news_articles"
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(COLLECTION_NAME)

def get_embeddings(texts):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"input": texts, "model": "jina-embeddings-v2-base-en"}
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return [d["embedding"] for d in data["data"]]

#Query
query = "latest news about AI"
query_emb = get_embeddings([query])[0]

results = collection.query(
    query_embeddings=[query_emb],
    n_results=5
)

print("Query:", query)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(meta["title"], "→", doc[:100], "...")

