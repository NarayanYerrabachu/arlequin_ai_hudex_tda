# HuDex TDA — Topological Document Analysis Engine

Pattern-intelligence engine that detects themes, anomalies, and entity relationships in document corpora — no LLMs, no training labels. Every finding traces back to source documents.

Runs at **`localhost:8002`** in the three-service HuDex stack.

---

## Where this fits

| | Prototype (8003) | Demo (8001) | **TDA (8002)** |
|---|---|---|---|
| Repo | `arlequin_ai_hudex_demo_prototype` (zip) | `arlequin_ai_hudex_demo_prototype` | **`arlequin_ai_hudex_tda`** |
| Corpus | Synthetic (random seed) | `corpus.csv` | **`corpus.csv`** (same) |
| Clustering | KMeans | KMeans + silhouette | **DBSCAN** |
| Anomaly signals | LOF + centroid + keyword | LOF + centroid + keyword | **IsoForest + Autoencoder + topo + keyword** |
| Per-doc topology | No | No | **Yes** |
| Graph storage | In-memory | In-memory | **Neo4j (in-memory fallback)** |
| Upload endpoint | No | Yes | **Yes** |
| Relationships | 14 (min\_shared=2) | 22 (min\_shared=1) | **22 (min\_shared=1)** |

The key differences vs Demo: DBSCAN allows noise docs (no forced cluster assignment), IsolationForest replaces LOF so clustered fraud communications are no longer treated as normal, the autoencoder adds a deep reconstruction-error signal, and per-document topology entropy is a fourth independent anomaly dimension.

See [`HuDex_NewCorpus_Comparison.md`](HuDex_NewCorpus_Comparison.md) for a side-by-side detection benchmark on a 104-document corpus with planted fraud patterns.

---

## Pipeline

```
Ingest
  └─ TF-IDF vectorize
       └─ DBSCAN clustering          (density-based themes, noise docs labelled -1)
       └─ IsolationForest            (statistical isolation score)
       └─ Autoencoder (PyTorch MLP)  (reconstruction-error outlier signal)
       └─ Per-doc topology entropy   (giotto-tda / ripser k-NN subgraph H1)
       └─ Corpus topology ripser     (H0 + H1 persistent homology fingerprint)
       └─ Entity extraction (regex)  (actors, money, accounts, dates, emails)
       └─ Co-occurrence graph
            └─ Neo4j bolt / in-memory fallback
```

### Stage details

| Stage | Method | Config |
|---|---|---|
| Vectorization | TF-IDF | `max_df=0.6`, `min_df=2`, `ngram_range=(1,2)`, English + name stopwords |
| Clustering | DBSCAN | eps swept `[0.35, 0.45, 0.55, 0.65]`, `min_samples=3`, cosine metric |
| Anomaly (signal 1) | IsolationForest | `n_estimators=200`, `contamination=0.08`, `random_state=7` |
| Anomaly (signal 2) | PyTorch Autoencoder | MLP dim→128→32→128→dim, 120 epochs, Adam lr=3e-3 |
| Anomaly (signal 3) | Keyword density | Hits against `SUSPICIOUS_VOCAB` in `patterns.yml` |
| Anomaly (signal 4) | Per-doc topology entropy | k=10 k-NN subgraph, VietorisRips H1, PersistenceEntropy; giotto-tda preferred, ripser fallback |
| Corpus topology | ripser H0+H1 | Sampled at most 60 docs, cosine distance matrix |
| Entity extraction | Regex (Unicode) | money, email, account, date, actor (TitleCase bigrams) |
| Relationships | Co-occurrence | `min_shared=1` document |
| Graph storage | Neo4j / in-memory | Bolt connection checked at startup; falls back silently |

### Anomaly score formula

```
score = 0.30 × IsolationForest
      + 0.25 × Autoencoder_reconstruction_error
      + 0.30 × keyword_density
      + 0.15 × local_topology_entropy
```

All four signals are min-max normalised before blending. A document scoring above the 8% contamination cutoff is classified as `anomaly`; if it also contains `SUSPICIOUS_VOCAB` hits it is reclassified as `suspicious`.

---

## DBSCAN vs KMeans

KMeans (Demo/Prototype) forces every document into a theme. DBSCAN does not:

| | KMeans | DBSCAN |
|---|---|---|
| Requires k upfront | Yes (swept by silhouette) | No |
| Noise / outlier docs | Assigned to nearest cluster | Labelled `-1` (unassigned) |
| Cluster shape | Spherical | Arbitrary density regions |
| On 112-doc corpus | 8 themes, 0 noise | 16 clusters, 23 noise docs |

Noise docs (`theme = -1`) are documents that do not belong to any dense region. They are not anomalies — the anomaly score is a separate signal. A noise doc can be completely normal (just unique in vocabulary) or can be genuinely suspicious.

---

## Per-document topology entropy

For each document, a k=10 nearest-neighbour subgraph is built from the cosine distance matrix. `VietorisRipsPersistence H1` is computed on that subgraph and `PersistenceEntropy` measures the Shannon entropy of the persistence barcode.

- **High entropy** = the document's local neighbourhood has complex, multi-scale topological structure — many competing loops of varying length.
- **Low entropy** = a clean, simple neighbourhood.

Deviation from the corpus median is used as the signal (absolute value). This is independent of vocabulary frequency (what IsolationForest measures) and independent of keyword presence, so it can surface structural anomalies that the other three signals miss.

### giotto-tda vs ripser fallback

| Environment | Backend | Notes |
|---|---|---|
| Linux amd64 (production ECS) | giotto-tda 0.6.2 | Parallelised C++ (`n_jobs=-1`) |
| Docker `--platform linux/amd64` | giotto-tda 0.6.2 | Emulation on Apple Silicon |
| Docker ARM64 (Mac native) | ripser (pure Python loop) | No ARM64 wheel for giotto-tda 0.6.2 |

Both paths compute identical mathematics. For production Docker builds:
```bash
docker build --platform linux/amd64 -t hudex-tda .
```

---

## Project structure

```
arlequin_ai_hudex_tda/
├── run.sh                           local pipenv start (port 8002)
├── run_all.sh                       also starts neo4j (for docker use)
├── docker-compose.yml               hudex-tda + neo4j containers
├── Dockerfile
├── Pipfile / Pipfile.lock
├── hudex_tda.html                   UI (served at /)
├── README.md
├── HuDex_Services_Documentation.md  full three-service reference
├── HuDex_NewCorpus_Comparison.md    detection benchmark (104 docs)
└── patternengine/
    ├── server.py                    FastAPI app + REST endpoints
    ├── corpus.csv                   default corpus (shared with Demo)
    ├── patterns.yml                 entity patterns + suspicious vocab
    ├── engine/
    │   ├── core_tda.py              full TDA pipeline (571 lines)
    │   └── sample_data.py           loads corpus.csv → Document list
    └── ingestion/
        └── loader.py                multi-format ingestion (CSV/PDF/DOCX/XLSX/JSON/TXT/XML)
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI (`hudex_tda.html`) |
| `GET` | `/api/findings` | Themes, anomalies, suspicious docs, relationships, graph, corpus TDA metadata |
| `GET` | `/api/documents` | All ingested documents with anomaly score and theme label |
| `POST` | `/api/query` | TF-IDF similarity search — body: `{"q": "...", "top": 8}` |
| `POST` | `/api/upload` | Replace corpus file; re-runs full pipeline; returns new findings inline |

### `/api/findings` response structure

```json
{
  "meta": {
    "n_docs": 112,
    "n_clusters": 16,
    "n_noise_docs": 23,
    "n_anomalies": 2,
    "n_suspicious": 7,
    "n_relationships": 22,
    "graph_backend": "in-memory",
    "torch_available": true,
    "ripser_available": true,
    "giotto_available": false,
    "neo4j_available": false,
    "tda": { "betti_0": 1, "betti_1": 29, "max_persistence": 0.352, "h1_features": [...] }
  },
  "themes": [...],
  "anomalies": [...],
  "suspicious": [...],
  "relationships": [...],
  "graph": { "nodes": [...], "edges": [...] }
}
```

### Finding object

```json
{
  "kind": "suspicious",
  "title": "Suspicious in DOC-F03",
  "detail": "Anomaly score 0.653. Keywords: no paperwork, shell entity",
  "score": 0.653,
  "sources": ["DOC-F03"],
  "extra": {
    "meta": { "channel": "comms_misc", "date": "2025-08-15" },
    "keywords": ["no paperwork", "shell entity"],
    "terms": ["shell", "entity", "paperwork", "offshore"]
  }
}
```

`extra.terms` are the TF-IDF terms the document over-uses relative to the corpus mean — data-derived explanation for why it was flagged, independent of the keyword list.

### Upload supported formats

`CSV`, `JSON`, `TXT`, `XML`, `PDF`, `DOCX`, `XLSX`

CSV must have an `id` column and a `text` column. Minimum 4 documents. State is replaced atomically (thread-safe lock).

---

## Running locally

```bash
# First time — generates Pipfile.lock and installs deps
./run.sh

# Service starts at http://localhost:8002
```

Or as part of the full three-service stack:
```bash
~/git/run_hudex_all.sh
# logs: tail -f /tmp/hudex_tda.log
```

---

## Running with Docker

```bash
docker-compose up
```

Starts two containers:
- `hudex-tda` on port **8002**
- `neo4j` on ports **7474** (browser UI) and **7687** (bolt)

With Neo4j running, relationships persist across restarts. Without it the engine starts in in-memory mode automatically.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection URI |
| `NEO4J_AUTH` | `neo4j/password` | Neo4j credentials (format: `user/password`) |
| `HUDEX_PATTERNS_FILE` | `patternengine/patterns.yml` | Path to custom patterns config |

---

## Requirements

- Python 3.11 (required for giotto-tda 0.6.2 wheel on amd64)
- pipenv
- Java 21+ (Apache Tika — PDF/DOCX ingestion fallback)
- Neo4j 5.x (optional)

---

## Known limitations

- **Meridian-ring style fraud** (repeated transfers to the same account across 6 individually unremarkable documents) is not detected. Each document is scored in isolation. Cross-document named-entity linking on account numbers with temporal sequence analysis is the next capability gap — see `HuDex_NewCorpus_Comparison.md` §Conclusions.
- **DBSCAN eps sensitivity** — the swept eps range `[0.35, 0.45, 0.55, 0.65]` works well on the current corpus size. At very small corpora (<30 docs) DBSCAN may assign most documents to noise; KMeans is more stable in that regime.
- **Autoencoder non-determinism** — PyTorch training uses `random_state=7` for IsolationForest but the autoencoder itself is not seeded. Anomaly scores will vary slightly across restarts. IsolationForest and keyword scores are fully deterministic.
- **Per-doc topology cost** — `_local_topo_scores` loops over every document and runs ripser on a k=10 subgraph. On 100 docs this takes ~2–5s; on 1000+ docs startup time increases linearly. giotto-tda with `n_jobs=-1` is significantly faster.
