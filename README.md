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

# Traffic Event-Aware Flow Prediction Framework

This is a joint traffic flow prediction demonstration framework that combines **event knowledge graph (KG) embedding**, **graph neural network (GNN)** and **time series transformer**.


## Core Modules

### 1. Event Knowledge Graph Construction and Embedding (TransE)

- Use event triples `(head, relation, tail)` to build event KG.

- Use a simple TransE model to map entities and relations to low-dimensional continuous vector space.

- After training, the embedding of entities and relations is obtained to represent the semantic information of events.

### 2. Event Vector Generation

- For a single event, sum or concatenate all the corresponding triple vectors to form a unified event vector representation.

- This vector is used as external semantic information to assist traffic flow prediction.

### 3. Graph Neural Network

- Use the characteristics of road sensor nodes and the road network topology (adjacency matrix) to build a graph structure.
- Use GCN to encode spatial dependencies and extract the spatial characteristics of traffic flow.

### 4. Time series dynamic prediction (Transformer)

- After splicing the time series traffic state encoded by GNN with the event vector, input it into the Transformer encoder.
- Transformer captures the complex dynamic changes in the time dimension.
- Finally predict the traffic flow at the future moment.

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

