# 🚦 Traffic Event Extractor

A LangChain-based demo to extract structured traffic events from news articles using OpenAI GPT.

## 📌 Features

- Crawl traffic news from a given URL
- Extract structured incident information (type, location, casualties, impact)
- Output clean JSON for downstream KG / prediction tasks

## 🏁 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
Create a `.env` file or use export:
```bash
export OPENAI_API_KEY=your-api-key
```

### 3. Run the extractor
```bash
python app.py
```

### ✅ Output Example

```json
{
  "event_type": "crash",
  "date": "2024-06-19",
  "location": "State Highway 1 near Seddon",
  "casualties": "1 dead",
  "impact": "road closed temporarily",
  "source_text": "One person has died following a single-vehicle crash..."
}
```

## 🔮 Future Work

- [ ] Multi-article batch processing
- [ ] Export to Neo4j for traffic knowledge graph
- [ ] Streamlit web UI
