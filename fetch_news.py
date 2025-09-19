from newsplease import NewsPlease
import json, uuid

urls = [
    "https://www.bbc.com/news/world-66798508",
    "https://edition.cnn.com/2024/07/15/business/tesla-stock-earnings/index.html",
    "https://www.reuters.com/world/asia-pacific/japan-economy-grows-2024-07-10/"
]

articles = []
for url in urls:
    try:
        article = NewsPlease.from_url(url)
        if article and article.maintext:
            articles.append({
                "id": str(uuid.uuid4()),
                "title": article.title,
                "url": url,
                "text": article.maintext
            })
    except Exception as e:
        print("Error:", url, e)

with open("news_articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"✅ Collected {len(articles)} articles")
