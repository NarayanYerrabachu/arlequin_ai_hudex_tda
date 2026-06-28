# HuDex TDA — Topological Document Analysis Engine

Pattern-intelligence engine that applies **Topological Data Analysis (TDA)** to document corpora. Detects themes, anomalies, and entity relationships using a multi-layer ML pipeline.

## Pipeline

```
Ingest → TF-IDF vectorize → Autoencoder (PyTorch) → Isolation Forest
       → DBSCAN clustering → Persistent Homology (ripser) → Entity graph → Neo4j
```

| Stage | Method | Purpose |
|---|---|---|
| Vectorization | TF-IDF | Sparse document embeddings |
| Anomaly scoring | PyTorch Autoencoder | Reconstruction-error outlier signal |
| Outlier detection | Isolation Forest | Statistical anomalies |
| Theme discovery | DBSCAN | Density-based clustering (no preset k) |
| Topology | ripser H1 | Persistent homology fingerprint per doc |
| Entity graph | Regex + co-occurrence | Relationship extraction |
| Graph storage | Neo4j (in-memory fallback) | Relationship persistence |

## Requirements

- Python 3.11
- pipenv
- Java 21 (for Apache Tika — PDF/DOCX ingestion)
- Neo4j (optional — falls back to in-memory if unavailable)

## Run locally

```bash
# First time — installs dependencies
./run.sh
```

Service starts at **http://localhost:8002**

## Run with Docker

```bash
docker-compose up
```

Starts two containers:
- `hudex-tda` on port **8002**
- `neo4j` on ports **7474** (browser UI) and **7687** (bolt)

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/findings` | Themes, anomalies, relationships, graph |
| `GET` | `/api/documents` | All ingested documents |
| `POST` | `/api/query` | Semantic search — body: `{"q": "...", "top": 8}` |
| `POST` | `/api/upload` | Upload new corpus file — replaces current state |

### Supported upload formats

`CSV`, `JSON`, `TXT`, `XML`, `PDF`, `DOCX`, `XLSX`

Minimum 4 documents required for meaningful analysis.

## Project structure

```
patternengine/
├── server.py           # FastAPI app + REST endpoints
├── engine/
│   ├── core_tda.py     # Full TDA pipeline (TF-IDF, autoencoder, DBSCAN, ripser, Neo4j)
│   └── sample_data.py  # Built-in demo corpus
└── ingestion/
    └── loader.py       # Multi-format document loader
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection URI |
| `NEO4J_AUTH` | `neo4j/password` | Neo4j credentials |

If Neo4j is unreachable the engine runs in in-memory mode — all analysis still works, relationships are not persisted across restarts.
