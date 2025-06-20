## Project Background

Traditional traffic flow prediction mostly relies on historical traffic data, ignoring the impact of emergencies (such as accidents, construction, and extreme weather) on traffic. This project constructs an event knowledge graph, extracts structured semantic information of events, combines the topological structure encoding of road network sensors, and then uses Transformer to capture dynamic changes in time series to achieve more robust and semantically rich traffic flow prediction.

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
Get the entities and relations of multiple events to form triples, such as:

```json
("crash", "occurred_on", "2024-06-19")
("crash", "caused", "road closed")
("crash", "happened_at", "State Highway 1 near Seddon")
```

# Traffic Event-Aware Flow Prediction Framework

This is a joint traffic flow prediction demonstration framework that combines **event knowledge graph (KG) embedding**, **graph neural network (GNN)** and **time series transformer**.


## Core Modules

 Framework Overview: Event KG + GNN + Transformer
This project proposes a novel traffic flow prediction framework by integrating event knowledge graphs, graph neural networks, and temporal Transformers.

🔗 Step 1: Event Knowledge Graph Construction
We extract structured semantic triples from traffic-related texts (e.g., news reports, social media) to build an event knowledge graph (KG).

📝 Example
From the news:

"One person died in a crash on State Highway 1 near Seddon, causing the road to be temporarily closed."

We extract:

triplets = [
    ("crash", "occurred_on", "2024-06-19"),
    ("crash", "location", "State Highway 1"),
    ("crash", "caused", "road_closure"),
    ("road_closure", "status", "temporary"),
    ("crash", "casualties", "1_dead")
]

🧊 Step 2: TransE Embedding of Event KG
We use the TransE model to embed each entity and relation into a low-dimensional vector space:

For each triple (h, r, t):

embedding(h) + embedding(r) ≈ embedding(t)

Example:

embedding("crash")       = [0.8, 0.1, -0.3, 0.2]
embedding("caused")      = [0.2, 0.3, -0.1, 0.1]
embedding("road_closure")= [1.0, 0.4, -0.4, 0.3]

We construct an event semantic vector by summing or averaging the embeddings of all related triples:

event_vec = mean([
    embed("crash") + embed("occurred_on") + embed("2024-06-19"),
    embed("crash") + embed("caused") + embed("road_closure"),
    ...
])

🛰️ Step 3: Traffic Flow Spatial Encoding (GNN)
We use real-world sensor data (e.g., METR-LA) to build a road network graph:

Nodes: traffic sensors

Edges: road connectivity

Each node has temporal features (e.g., speed, flow), and is encoded using GCN (Graph Convolutional Network):

GNN_output_t1, GNN_output_t2, ..., GNN_output_tn

⏳ Step 4: Temporal Prediction (Transformer) + Event Fusion
We concatenate the event vector with each time step's GCN output:

input_t1 = concat(GNN_output_t1, event_vec)
input_t2 = concat(GNN_output_t2, event_vec)
...
input_tn = concat(GNN_output_tn, event_vec)

Then feed into a Transformer for time series modeling:

prediction = Transformer([input_t1, ..., input_tn])

🔁 Final Output
The Transformer outputs the predicted traffic flow (speed / volume) for future time steps, with enhanced robustness to unexpected events.
---

## Project Structure

traffic-event-prediction/
├── data/ # Store traffic flow data, event triples and road network structure

├── kg/ # Event knowledge graph construction and TransE embedding

│ ├── transE.py # TransE model implementation

│ ├── utils.py # Entity relationship mapping and tool functions

│ ├── embed_event.py # Event vector generation

├── model/ # GNN and Transformer model

│ ├── gnn.py # Road network graph neural network

│ ├── transformer.py # Transformer encoder module

│ ├── predictor.py # Joint prediction model (event embedding + GNN + Transformer)

├── train.py # Main training script

├── evaluate.py # Model evaluation script

├── requirements.txt # Dependency package list

├── README.md # Project description document

