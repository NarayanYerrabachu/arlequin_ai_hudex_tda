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

### Local (pipenv)
```bash
# First time — installs deps
chmod +x ~/git/arlequin_ai_hudex_tda/run.sh \
         ~/git/arlequin_ai_hudex_demo_prototype/run.sh \
         ~/git/arlequin_ai_hudex_demo_prototype/run_prototype.sh \
         ~/git/run_hudex_all.sh

# Run all three
~/git/run_hudex_all.sh

# Or individually
~/git/arlequin_ai_hudex_demo_prototype/run_prototype.sh   # 8003
~/git/arlequin_ai_hudex_demo_prototype/run.sh             # 8001
~/git/arlequin_ai_hudex_tda/run.sh                        # 8002
```

### Docker
```bash
cd ~/git/arlequin_ai_hudex_demo_prototype && docker-compose up   # 8001
cd ~/git/arlequin_ai_hudex_tda && docker-compose up             # 8002 + Neo4j
```

### Logs
```bash
tail -f /tmp/hudex_demo.log    # 8001
tail -f /tmp/hudex_tda.log     # 8002
tail -f /tmp/hudex_proto.log   # 8003
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
```
