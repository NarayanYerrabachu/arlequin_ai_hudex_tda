# HuDex — Service Documentation

Three pattern-intelligence engines that analyse document corpora offline, without LLMs or training labels. Every finding is traceable back to source documents.

---

## Overview

| | Prototype | Demo | TDA |
|---|---|---|---|
| **Port** | 8003 | 8001 | 8002 |
| **Repo** | `arlequin_ai_hudex_demo_prototype` (zip) | `arlequin_ai_hudex_demo_prototype` | `arlequin_ai_hudex_tda` |
| **Engine** | `PatternEngine` (v0) | `PatternEngine` (v1) | `TDAPatternEngine` |
| **Clustering** | KMeans | KMeans + silhouette | DBSCAN |
| **Anomaly detection** | LOF + centroid + keyword | LOF + centroid + keyword | IsolationForest + Autoencoder + topology + keyword |
| **Topology (TDA)** | ripser (corpus-level) | ripser (corpus-level) | giotto-tda + ripser (per-doc + corpus-level) |
| **Graph storage** | In-memory | In-memory | Neo4j (in-memory fallback) |
| **Suspicious vocab** | Yes | Yes | Yes |
| **Upload endpoint** | No | Yes | Yes |
| **Relationships** | 14 (min\_shared=2) | 22 (min\_shared=1) | 22 (min\_shared=1) |

---

## Service 1 — Prototype (`localhost:8003`)

The original proof-of-concept. Simplest pipeline, no server originally — `export.py` wrote `findings.json` for a static HTML viewer. Now wrapped in a minimal FastAPI server.

### Pipeline

```
Ingest → TF-IDF → KMeans → IsolationForest → ripser → Entities → Relationships
```

### Algorithms

| Stage | Method | Detail |
|---|---|---|
| Vectorization | TF-IDF | `stop_words="english"`, `max_df=0.6`, `min_df=2`, `ngram_range=(1,2)` |
| Clustering | KMeans | k selected by silhouette score over `range(3,9)` |
| Anomaly detection | LOF + centroid distance + keyword density | 3-signal ensemble (same as Demo) |
| Suspicious detection | Keyword match | `SUSPICIOUS_VOCAB` from `patterns.yml` |
| Topology | ripser H0+H1 | Corpus-level persistent homology fingerprint |
| Entity extraction | Regex | money, email, account, date, actor (TitleCase bigrams) |
| Relationships | Co-occurrence | `min_shared=2` documents |

### Anomaly score formula
```
score = 0.6 × LOF + 0.4 × centroid_distance
```

### Current stats (112 docs)
- Themes: **8** | Theme quality: 0.187
- Anomalies: **4** | Suspicious: **5**
- Relationships: **14**
- TDA: Betti₀=1, Betti₁=48, max persistence=0.2252

### Limitations
- `min_shared=2` produces fewer relationships than v1/TDA
- No upload endpoint — corpus is fixed at startup
- No per-document topology entropy (corpus-level only)
- No topology signal in anomaly scoring

---

## Service 2 — Demo (`localhost:8001`)

Upgraded prototype with a full FastAPI server, improved anomaly detection using two complementary signals, and per-document topological fingerprint.

### Pipeline

```
Ingest → TF-IDF → KMeans → LOF + Centroid + Keyword → ripser → Entities → Relationships
```

### Algorithms

| Stage | Method | Detail |
|---|---|---|
| Vectorization | TF-IDF | Name stopwords added, `max_df=0.6`, `min_df=2`, `ngram_range=(1,2)` |
| Clustering | KMeans | k selected by silhouette score over `range(3,9)` |
| Anomaly detection | LOF + Centroid distance + Keyword density | 3-signal ensemble |
| Suspicious detection | Keyword match | `SUSPICIOUS_VOCAB` from `patterns.yml` |
| Topology | ripser H0+H1 | Corpus-level persistent homology fingerprint |
| Entity extraction | Regex (Unicode) | Handles German umlauts (Müller, Schäfer) |
| Relationships | Co-occurrence | `min_shared=1` document |

### Anomaly score formula
```
score = 0.5 × LOF + 0.3 × centroid_distance + 0.2 × keyword_density
```
LOF catches hidden clusters; centroid distance catches rare vocabulary; keyword density surfaces financial crime language.

### Current stats (112 docs)
- Themes: **8** | Theme quality: 0.251
- Anomalies: **1** | Suspicious: **8**
- Relationships: **22**
- TDA: Betti₀=1, Betti₁=29, max persistence=0.352

### Improvements over Prototype
- Keyword density added as 3rd anomaly signal
- `min_shared=1` captures more relationships
- Upload endpoint — corpus can be replaced at runtime
- Name stopwords prevent person names dominating themes

---

## Service 3 — TDA Engine (`localhost:8002`)

Full research-grade pipeline. Adds a PyTorch autoencoder for deep anomaly detection, per-document topological scoring via ripser, DBSCAN for density-based clustering, and Neo4j for relationship persistence.

### Pipeline

```
Ingest → TF-IDF → DBSCAN → IsolationForest + Autoencoder + Topology + Keyword
       → ripser (per-doc + corpus) → Entities → Neo4j / in-memory
```

### Algorithms

| Stage | Method | Detail |
|---|---|---|
| Vectorization | TF-IDF | Same as Demo |
| Clustering | DBSCAN | eps swept over `[0.35, 0.45, 0.55, 0.65]`, `min_samples=3`, cosine metric |
| Anomaly detection | IsolationForest + Autoencoder + Topology + Keyword | 4-signal ensemble |
| Autoencoder | PyTorch MLP | dim→128→32→128→dim, 120 epochs, Adam lr=3e-3 |
| Per-doc topology | giotto-tda VietorisRipsPersistence + PersistenceEntropy | k=10 k-NN subgraph per doc; ripser fallback on ARM64 |
| Corpus topology | ripser H0+H1 | Same as Demo |
| Suspicious detection | Keyword match | `SUSPICIOUS_VOCAB` from `patterns.yml` |
| Entity extraction | Regex (Unicode) | Same as Demo |
| Relationships | Co-occurrence | `min_shared=1` document |
| Graph storage | Neo4j bolt | Falls back to in-memory if Neo4j unreachable |
| Corpus shape | KeplerMapper Mapper algorithm | Interactive HTML at `/api/topology-map` |
| Drift detection | GUDHI bottleneck + Wasserstein distance | Compares H1 diagrams across uploads |

### Anomaly score formula
```
score = 0.30 × IsolationForest + 0.25 × Autoencoder_reconstruction_error
      + 0.30 × keyword_density + 0.15 × local_topology_entropy
```

### DBSCAN vs KMeans

| | KMeans (Prototype/Demo) | DBSCAN (TDA) |
|---|---|---|
| Requires k upfront | Yes (swept by silhouette) | No |
| Handles noise docs | No (every doc assigned) | Yes (noise = cluster -1) |
| Cluster shape | Spherical | Arbitrary |
| Result | 8 themes | 16 clusters + 23 noise docs |

### Additional features (TDA only)

**Corpus Map tab** — KeplerMapper renders the document corpus as an interactive graph. Nodes are clusters of similar documents; edges link overlapping neighbourhoods. Loaded lazily on first tab click from `/api/topology-map`.

**Drift detection** — every upload computes the topological distance between the previous corpus and the new one using GUDHI Wasserstein distance on H1 persistence diagrams. The previous corpus becomes the new baseline after each upload.

| Drift label | Wasserstein score | Meaning |
|---|---|---|
| None | < 0.05 | Corpus shape unchanged |
| Minimal | < 0.15 | Small structural shift |
| Moderate | < 0.30 | Noticeable shape change |
| Significant | < 0.50 | Major structural change |
| Major | ≥ 0.50 | Corpus fundamentally different |

On ARM64 (no GUDHI wheel), drift falls back to mean-persistence difference.

### REST API — TDA-only endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/topology-map` | KeplerMapper interactive HTML corpus graph |

Upload response also includes a `drift` object: `{bottleneck, wasserstein, label, gudhi_available}`.

### Current stats (112 docs)
- Clusters: **16** | Noise docs: **23**
- Anomalies: **2** | Suspicious: **7**
- Relationships: **22**
- TDA: Betti₀=1, Betti₁=29, max persistence=0.352
- Torch: available | ripser: available | Neo4j: in-memory fallback

### giotto-tda vs ripser fallback

| Environment | Backend | Notes |
|---|---|---|
| Linux amd64 (production) | giotto-tda 0.6.2 | Pre-built wheel — `_GIOTTO=True` |
| Docker `--platform linux/amd64` | giotto-tda 0.6.2 | Same wheel via emulation |
| Docker ARM64 (Mac Apple Silicon) | ripser fallback | No ARM64 wheel — `_GIOTTO=False` |

Both paths compute `VietorisRipsPersistence H1 + PersistenceEntropy` on each document's k-NN subgraph — the math is identical. giotto-tda uses parallelised C++ extensions (`n_jobs=-1`); the ripser path loops in Python.

Production Docker build:
```bash
docker build --platform linux/amd64 -t hudex-tda .
```

### Improvements over Demo
- Autoencoder adds deep representation anomaly signal
- Per-document topology entropy via giotto-tda — each doc gets its own neighbourhood complexity score
- DBSCAN — no forced cluster assignment; noisy/unique docs left unclassified
- Neo4j — relationships persist across restarts (when running)

---

## TDA Libraries

### Ripser.py
Fast, lean library in terms of memory and compute for computing Vietoris-Rips persistent homology. The workhorse for all corpus-level and per-document H0/H1 diagrams across all three services. Falls back from giotto-tda on ARM64.

### giotto-tda
High-performance library designed to integrate TDA with machine learning pipelines (scikit-learn API compatible). Used in the TDA engine for per-document `VietorisRipsPersistence + PersistenceEntropy` on each document's k-NN subgraph. Runs parallelised C++ extensions (`n_jobs=-1`) on amd64; ripser fallback on ARM64 where no wheel is available.

### KeplerMapper
Go-to Python implementation of the Mapper algorithm, offering interactive HTML visualisations of data shapes. Used in the TDA engine's Corpus Map tab (`/api/topology-map`) to render the document corpus as a navigable graph — nodes are document clusters, edges connect overlapping neighbourhoods. Installed with `--no-deps` to avoid the openmp build conflict on ARM64.

### GUDHI
Robust C++/Python library backed by INRIA for computing persistent homology and geometric complexes. Used for drift detection in the TDA engine: bottleneck distance and Wasserstein distance compare H1 persistence diagrams between the previous corpus and a newly uploaded one. No prebuilt wheel for linux/arm64 — falls back to mean-persistence comparison when unavailable.

---

## Technology Stack

| Layer | Technology | Used by |
|---|---|---|
| Web framework | FastAPI | All 3 |
| ASGI server | Uvicorn | All 3 |
| Vectorization | scikit-learn TF-IDF | All 3 |
| Clustering | scikit-learn KMeans | Prototype, Demo |
| Clustering | scikit-learn DBSCAN | TDA |
| Anomaly detection | scikit-learn IsolationForest | Prototype, TDA |
| Anomaly detection | scikit-learn LocalOutlierFactor | Demo |
| Anomaly detection | PyTorch Autoencoder (MLP) | TDA |
| Persistent homology | ripser | All 3 |
| Per-doc topology pipeline | giotto-tda 0.6.2 | TDA (amd64); ripser fallback on ARM64 |
| Corpus shape visualisation | KeplerMapper | TDA |
| Drift detection | GUDHI + POT | TDA (mean-persistence fallback on ARM64) |
| Document ingestion | Apache Tika (JVM) | All 3 |
| PDF fallback | pdfplumber | All 3 |
| XLSX ingestion | pandas + openpyxl | All 3 |
| Graph database | Neo4j (bolt) | TDA |
| Pattern config | PyYAML (`patterns.yml`) | All 3 |
| Runtime | Python 3.13 | Prototype, Demo |
| Runtime | Python 3.11 | TDA (required for giotto-tda 0.6.2 wheel) |
| Dependency mgmt | pipenv | All 3 |
| Containerisation | Docker + docker-compose | All 3 |

---

## Shared Features (All 3 Services)

### External pattern configuration
All engines load `ENTITY_PATTERNS` and `SUSPICIOUS_VOCAB` from `patternengine/patterns.yml` at startup. The file location can be overridden per deployment:
```bash
HUDEX_PATTERNS_FILE=/etc/customer_xyz/patterns.yml ./run.sh
```
Falls back to built-in defaults if the file is absent or `pyyaml` is not installed.

### Entity types
| Type | Pattern | Example |
|---|---|---|
| `money` | Currency + amount | `EUR 80,000` |
| `email` | RFC-style email | `user@domain.com` |
| `account` | ACC/IBAN/REF prefix | `ACC-X4829` |
| `date` | ISO 8601 | `2024-03-15` |
| `actor` | TitleCase bigram (Unicode) | `Anna Müller` |

### Suspicious vocabulary
Loaded from `patterns.yml`. Covers financial crime language: `off the books`, `kickback`, `shell company`, `launder`, `untraceable`, `offshore`, etc. (24 terms by default).

### Graph node colouring
| Colour | Meaning |
|---|---|
| Red | Actor appears in suspicious documents |
| Amber | Actor appears in anomaly documents (no suspicious exposure) |
| Dark | Actor appears only in clean documents |

### REST API (all services)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/findings` | Themes, anomalies, suspicious, relationships, graph |
| `GET` | `/api/documents` | All ingested documents with scores |
| `POST` | `/api/query` | TF-IDF similarity search — `{"q": "...", "top": 8}` |
| `POST` | `/api/upload` | Replace corpus — CSV, JSON, TXT, XML, PDF, DOCX, XLSX |

*(Upload not available on Prototype)*

---

## Running the Services

All three services run in Docker. Use the `hudex.sh` script at `~/git/hudex.sh`:

```bash
# First time — make executable
chmod +x ~/git/hudex.sh
```

```bash
~/git/hudex.sh start          # build + start all 3 services
~/git/hudex.sh stop           # stop all
~/git/hudex.sh status         # show running containers
~/git/hudex.sh logs 8001      # tail logs for demo
~/git/hudex.sh logs 8002      # tail logs for TDA engine
~/git/hudex.sh logs 8003      # tail logs for prototype
```

| URL | Service |
|---|---|
| http://localhost:8001 | HuDex Demo |
| http://localhost:8002 | HuDex TDA |
| http://localhost:8003 | HuDex Prototype |

### Docker compose files
| Service | Compose file |
|---|---|
| 8002 + Neo4j | `arlequin_ai_hudex_tda/docker-compose.yml` |
| 8001 + 8003 | `arlequin_ai_hudex_demo_prototype/docker-compose.yml` |

### Test corpus
```
arlequin_ai_hudex_tda/patternengine/corpus.csv          # for 8002
arlequin_ai_hudex_demo_prototype/patternengine/corpus.csv  # for 8001 / 8003
```

---

## Evolution Path

```
Prototype (8003)
  └─ Fixed corpus, LOF + centroid anomaly, corpus TDA (ripser), suspicious vocab
      │
      ▼
Demo (8001)
  └─ Live server, upload support, corpus TDA, name stopwords, min_shared=1
      │
      ▼
TDA Engine (8002)
  └─ Autoencoder, per-doc topology entropy, DBSCAN, Neo4j, 4-signal anomaly scoring
  └─ KeplerMapper: interactive corpus shape visualisation (Corpus Map tab)
  └─ GUDHI: drift detection — Wasserstein distance between H1 diagrams on each upload
```
