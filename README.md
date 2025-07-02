## Project Background

Traditional traffic flow prediction mostly relies on historical traffic data, ignoring the impact of emergencies (such as accidents, construction, and extreme weather) on traffic. This project constructs an event knowledge graph, extracts structured semantic information of events, combines the topological structure encoding of road network sensors, and then uses Transformer to capture dynamic changes in time series to achieve more robust and semantically rich traffic flow prediction.

# 🚦 Traffic Event Extractor

A LangChain-based demo to extract structured traffic events from news articles using OpenAI GPT.

## 📌 
🔧 技术栈
💬 大语言模型：OpenAI GPT-4


🔗 框架：LangChain（Prompt 管理 + LLMChain + OutputParser）

🧱 结构化输出：JSON → 三元组

📊 下游任务：事件知识图谱构建 / GNN-Transformer 联合建模

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

### 3.  Prompt 示例
```bash
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a traffic event extraction assistant. Please extract structured traffic event information from the following text and return it in JSON format:

{text}

Output fields include:
- event_type (e.g., crash, road closure, etc.)
- date (date of occurrence)
- location (location)
- impact (e.g., road closure, delays)
- casualties (number of casualties)

If any fields are missing, return an empty value.
"""
)

"""
)

```
### ✅ LLMChain 调用示例
```bash
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
chain = LLMChain(prompt=prompt, llm=llm)

text = "One person has died in a single-vehicle crash on State Highway 1 near Seddon. The road is closed until further notice."
result = chain.run(text=text)
print(result)
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



## 🧠 Framework Overview: Event KG + GNN + Transformer

This project proposes a novel traffic flow prediction framework by integrating **event knowledge graphs**, **graph neural networks**, and **temporal Transformers**. It aims to enhance the robustness and interpretability of traffic prediction in the presence of unexpected events such as accidents, road closures, or extreme weather.

---

### 🔗 Step 1: Event Knowledge Graph Construction

We extract structured semantic triples from traffic-related texts (e.g., news reports, social media) to build an event knowledge graph (KG).

#### 📝 Example

From the news:
> "One person died in a crash on State Highway 1 near Seddon, causing the road to be temporarily closed."

We extract the following triples:

```python
triplets = [
    ("crash", "occurred_on", "2024-06-19"),
    ("crash", "location", "State Highway 1"),
    ("crash", "caused", "road_closure"),
    ("road_closure", "status", "temporary"),
    ("crash", "casualties", "1_dead")
]
```

---

### 🧊 Step 2: TransE Embedding of Event KG

We use the **TransE** model to embed each entity and relation into a continuous vector space.

> For each triple `(h, r, t)`, TransE aims to learn:
> ```math
>     embedding(h) + embedding(r) ≈ embedding(t)
> ```

Assume the simplified embeddings:

```text
embedding("crash")        = [0.8, 0.1, -0.3, 0.2]
embedding("caused")       = [0.2, 0.3, -0.1, 0.1]
embedding("road_closure") = [1.0, 0.4, -0.4, 0.3]
```

We compute an **event semantic vector** by aggregating over all triples:

```python
event_vector_i = embedding(h) + embedding(r) + embedding(t)
event_vec = mean([event_vector_1, event_vector_2, ..., event_vector_n])
```

This `event_vec` captures the semantic meaning of the event and its potential impact on traffic.

---

### 🛰️ Step 3: Traffic Flow Spatial Encoding (GNN)

We use real-world traffic sensor data (e.g., from METR-LA) and model the road network as a graph:

- **Nodes**: traffic sensors
- **Edges**: adjacency based on road topology
- **Features**: traffic speed, volume, and occupancy at each time step

We apply a **GCN (Graph Convolutional Network)** to model spatial dependencies:

```python
GNN_output_t1, GNN_output_t2, ..., GNN_output_tn
```

Each `GNN_output_tx` is a feature embedding for all nodes at time `t`.

---

### ⏳ Step 4: Temporal Prediction with Transformer + Event Fusion

We **fuse event information** with GCN spatial features by concatenation:

```python
input_t1 = concat(GNN_output_t1, event_vec)
input_t2 = concat(GNN_output_t2, event_vec)
...
input_tn = concat(GNN_output_tn, event_vec)
```

We feed the sequence into a **Transformer** to capture temporal dependencies:

```python
prediction = Transformer([input_t1, input_t2, ..., input_tn])
```

---

### 🔮 Output

The Transformer outputs predicted traffic conditions (speed/flow) at future time points. By including real-world event information, the model can adapt to emergencies and provide more reliable forecasts.

---

### 📌 Text-based Framework Diagram

```
[Text] → [Event Triples] → [Event KG] → [TransE] → 🧩 event_vec
                                         ↓
[Sensor Graph + Features] → [GNN] → [GNN_output_t1, ..., tn]
                                         ↓
   [GNN_output_tx + event_vec] → [Transformer] → [Predicted Flow Values]
```

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

