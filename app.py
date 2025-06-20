from chains.event_extraction_chain import event_chain
from utils.url_loader import extract_article_text_from_url
import json
import os

url = "https://www.stuff.co.nz/nz-news/360728891/one-dead-after-single-vehicle-crash-state-highway-1-near-seddon"
article = extract_article_text_from_url(url)
print("🔍 Article Content:\n", article[:300], "...")

result = event_chain.run(article=article)

os.makedirs("output", exist_ok=True)
with open("output/structured_event.json", "w") as f:
    f.write(result)

print("✅ Extraction Result:\n", result)
