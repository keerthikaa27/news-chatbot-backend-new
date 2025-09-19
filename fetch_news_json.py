import pandas as pd
import uuid
import json
import random

csv_path = r"C:\Users\keert\OneDrive\Documents\Rag-chatbot\train.csv"

df = pd.read_csv(csv_path, header=None)
df.columns = ["Class", "Title", "Description"]  # dataset format

sampled = df.sample(50, random_state=42)

articles = []
for _, row in sampled.iterrows():
    articles.append({
        "id": str(uuid.uuid4()),
        "title": row["Title"],
        "url": "N/A",  # dataset doesn’t have URLs
        "text": row["Description"]
    })

# Save as JSON
with open("news_articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"Saved {len(articles)} articles to news_articles.json")
