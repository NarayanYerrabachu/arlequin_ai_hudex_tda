"""
TDA Pattern-Intelligence Engine — report-specified full stack.

Pipeline:
  ingest
  -> vectorize (TF-IDF, same as baseline)
  -> autoencoder (PyTorch) for reconstruction-error anomaly scoring
  -> Isolation Forest for statistical outlier detection
  -> DBSCAN for density-based theme discovery (no need to pre-specify k)
  -> persistent homology (ripser) for topological structure fingerprint
  -> entity extraction (regex) + co-occurrence graph
  -> Neo4j graph DB for relationship storage (falls back to in-memory if unavailable)
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

from engine.patterns_loader import load_patterns

# --------------------------------------------------------------------------
# Optional deep-learning autoencoder (PyTorch)
# --------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

# --------------------------------------------------------------------------
# Persistent homology — giotto-tda preferred, ripser fallback
# --------------------------------------------------------------------------
try:
    from ripser import ripser
    _RIPSER = True
except ImportError:
    _RIPSER = False

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy as GiottoPersistenceEntropy
    _GIOTTO = True
except ImportError:
    _GIOTTO = False

try:
    import gudhi
    import gudhi.wasserstein
    _GUDHI = True
except ImportError:
    _GUDHI = False

try:
    import kmapper as km
    _KMAPPER = True
except ImportError:
    _KMAPPER = False

# --------------------------------------------------------------------------
# Topology helpers
# --------------------------------------------------------------------------

def _persistence_entropy(h1_finite: np.ndarray) -> float:
    """Persistence entropy of an H1 diagram (ripser fallback path)."""
    if len(h1_finite) == 0:
        return 0.0
    pers = h1_finite[:, 1] - h1_finite[:, 0]
    total = pers.sum()
    if total < 1e-12:
        return 0.0
    p = pers / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def _local_topo_scores(Xd: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-document topology anomaly signal via VietorisRipsPersistence H1 +
    PersistenceEntropy.  Uses giotto-tda when available, falls back to ripser.

    For each document: build its k-NN distance subgraph, compute H1 persistence
    entropy, return each doc's deviation from the corpus median.
    High deviation = anomalously complex local topology.
    """
    if len(Xd) < k + 2:
        return np.zeros(len(Xd))

    from sklearn.metrics.pairwise import cosine_distances
    dist = cosine_distances(Xd)

    if _GIOTTO:
        return _local_topo_scores_giotto(Xd, dist, k)
    elif _RIPSER:
        return _local_topo_scores_ripser(dist, k)
    return np.zeros(len(Xd))


def _local_topo_scores_giotto(Xd: np.ndarray, dist: np.ndarray, k: int) -> np.ndarray:
    """giotto-tda path: VietorisRipsPersistence + PersistenceEntropy per k-NN subgraph."""
    vrp = VietorisRipsPersistence(homology_dimensions=[1], metric="precomputed", n_jobs=-1)
    pe  = GiottoPersistenceEntropy()
    entropies = np.zeros(len(Xd))
    for i in range(len(Xd)):
        neighbours = np.argsort(dist[i])[1 : k + 1]
        idx = np.concatenate([[i], neighbours])
        sub = dist[np.ix_(idx, idx)][np.newaxis]   # shape (1, n, n)
        try:
            diagrams   = vrp.fit_transform(sub)    # shape (1, n_bars, 3)
            entropy    = pe.fit_transform(diagrams) # shape (1, n_dims)
            entropies[i] = float(entropy[0, 0])    # H1 entropy
        except Exception:
            entropies[i] = 0.0
    median = np.median(entropies)
    return np.abs(entropies - median)


def _local_topo_scores_ripser(dist: np.ndarray, k: int) -> np.ndarray:
    """ripser fallback path — same maths as giotto-tda, implemented manually."""
    entropies = np.zeros(len(dist))
    for i in range(len(dist)):
        neighbours = np.argsort(dist[i])[1 : k + 1]
        idx = np.concatenate([[i], neighbours])
        sub = dist[np.ix_(idx, idx)]
        try:
            result = ripser(sub, maxdim=1, distance_matrix=True)
            h1 = result["dgms"][1] if len(result["dgms"]) > 1 else np.empty((0, 2))
            finite_h1 = h1[h1[:, 1] < np.inf]
            entropies[i] = _persistence_entropy(finite_h1)
        except Exception:
            entropies[i] = 0.0
    median = np.median(entropies)
    return np.abs(entropies - median)


# --------------------------------------------------------------------------
# Neo4j graph DB
# --------------------------------------------------------------------------
try:
    from neo4j import GraphDatabase
    _NEO4J = True
except ImportError:
    _NEO4J = False

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ("neo4j", "password")


# --------------------------------------------------------------------------
# Data model (identical contract to baseline engine)
# --------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


# Loaded from patternengine/patterns.yml (or built-in defaults if file absent).
# Set HUDEX_PATTERNS_FILE=/path/to/patterns.yml to override at runtime.
ENTITY_PATTERNS, SUSPICIOUS_VOCAB = load_patterns()


@dataclass
class Finding:
    kind: str
    title: str
    detail: str
    score: float
    sources: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Autoencoder (simple MLP, runs on CPU — no GPU needed for small corpus)
# --------------------------------------------------------------------------

class _Autoencoder(nn.Module):  # type: ignore[misc]
    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, bottleneck), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def _train_autoencoder(Xd: np.ndarray, epochs: int = 120) -> np.ndarray:
    """Train autoencoder; return per-document reconstruction error (MSE)."""
    if not _TORCH:
        return np.zeros(len(Xd))
    X = torch.tensor(Xd, dtype=torch.float32)
    model = _Autoencoder(Xd.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        recon = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        recon = model(X).numpy()
    errors = ((Xd - recon) ** 2).mean(axis=1)
    return errors


# --------------------------------------------------------------------------
# Topological fingerprint via ripser
# --------------------------------------------------------------------------

def _tda_fingerprint(Xd: np.ndarray) -> dict[str, Any]:
    """Compute H0 and H1 persistent homology on the full doc-feature matrix.
    Returns Betti numbers and the five most persistent H1 features."""
    if not _RIPSER:
        return {"betti_0": None, "betti_1": None, "max_persistence": None,
                "h1_features": [], "available": False}

    # ripser works on a distance matrix; sample at most 60 docs to keep it cheap
    sample = Xd if len(Xd) <= 60 else Xd[np.random.default_rng(42).choice(len(Xd), 60, replace=False)]
    result = ripser(sample, maxdim=1, metric="cosine")

    dgms = result["dgms"]
    # H0: connected components (alive at distance 0)
    h0 = dgms[0]
    betti_0 = int((h0[:, 1] == np.inf).sum()) if len(h0) else 1

    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    finite_h1 = h1[h1[:, 1] < np.inf]
    persistence = (finite_h1[:, 1] - finite_h1[:, 0]) if len(finite_h1) else np.array([])
    betti_1 = len(finite_h1)
    max_pers = float(persistence.max()) if len(persistence) else 0.0

    top5 = sorted(zip(finite_h1[:, 0].tolist(), finite_h1[:, 1].tolist()),
                  key=lambda x: x[1]-x[0], reverse=True)[:5]

    return {
        "betti_0": betti_0,
        "betti_1": betti_1,
        "max_persistence": round(max_pers, 4),
        "h1_features": [{"birth": round(b, 4), "death": round(d, 4), "persistence": round(d-b, 4)}
                        for b, d in top5],
        "available": True,
    }


# --------------------------------------------------------------------------
# Neo4j relationship storage
# --------------------------------------------------------------------------

class _GraphStore:
    """Write actor relationships to Neo4j; fall back to in-memory dict."""

    def __init__(self):
        self._driver = None
        self._inmem: list[dict] = []
        if _NEO4J:
            try:
                drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
                drv.verify_connectivity()
                self._driver = drv
            except Exception:
                self._driver = None

    @property
    def backend(self) -> str:
        return "neo4j" if self._driver else "in-memory"

    def store(self, rels: list[dict]):
        self._inmem = rels
        if not self._driver:
            return
        with self._driver.session() as s:
            s.run("MATCH (n:Actor) DETACH DELETE n")
            for r in rels:
                s.run(
                    "MERGE (a:Actor {name:$a}) MERGE (b:Actor {name:$b}) "
                    "MERGE (a)-[e:CO_OCCURS {weight:$w}]->(b)",
                    a=r["a"], b=r["b"], w=r["weight"],
                )

    def load(self) -> list[dict]:
        if not self._driver:
            return self._inmem
        with self._driver.session() as s:
            rows = s.run(
                "MATCH (a:Actor)-[e:CO_OCCURS]->(b:Actor) "
                "RETURN a.name AS a, b.name AS b, e.weight AS w"
            )
            return [{"a": r["a"], "b": r["b"], "weight": r["w"]} for r in rows]

    def close(self):
        if self._driver:
            self._driver.close()


# --------------------------------------------------------------------------
# Drift detection (GUDHI Wasserstein / Bottleneck distance)
# --------------------------------------------------------------------------

def extract_h1_diagram(Xd: np.ndarray) -> np.ndarray:
    """Compute the H1 persistence diagram for a document matrix.
    Returns array of shape (n_bars, 2) with [birth, death] rows.
    Uses ripser if available, else returns empty array.
    """
    if not _RIPSER:
        return np.empty((0, 2))
    sample = Xd if len(Xd) <= 80 else Xd[
        np.random.default_rng(42).choice(len(Xd), 80, replace=False)
    ]
    result = ripser(sample, maxdim=1, metric="cosine")
    h1 = result["dgms"][1] if len(result["dgms"]) > 1 else np.empty((0, 2))
    # remove infinite bars — GUDHI distance requires finite diagrams
    return h1[h1[:, 1] < np.inf]


def compute_drift(baseline_dgm: np.ndarray, new_dgm: np.ndarray) -> dict:
    """Compare two H1 persistence diagrams using GUDHI.

    Returns bottleneck distance, Wasserstein distance, and a human label.
    Falls back to a simple persistence-vector comparison if GUDHI unavailable.
    """
    if not _GUDHI:
        # simple fallback: compare mean persistence
        def mean_pers(d): return float((d[:, 1] - d[:, 0]).mean()) if len(d) else 0.0
        diff = abs(mean_pers(new_dgm) - mean_pers(baseline_dgm))
        return {"bottleneck": round(diff, 4), "wasserstein": round(diff, 4),
                "label": _drift_label(diff), "gudhi_available": False}

    # GUDHI expects list of (birth, death) tuples; empty diagram = [(0,0)]
    def to_gudhi(d):
        return d.tolist() if len(d) else [[0.0, 0.0]]

    bottleneck = gudhi.bottleneck_distance(to_gudhi(baseline_dgm), to_gudhi(new_dgm))
    wasserstein = gudhi.wasserstein.wasserstein_distance(
        np.array(to_gudhi(baseline_dgm)),
        np.array(to_gudhi(new_dgm)),
        order=1, internal_p=2,
    )
    return {
        "bottleneck": round(float(bottleneck), 4),
        "wasserstein": round(float(wasserstein), 4),
        "label": _drift_label(float(wasserstein)),
        "gudhi_available": True,
    }


def _drift_label(score: float) -> str:
    if score < 0.05:  return "none"
    if score < 0.15:  return "minimal"
    if score < 0.30:  return "moderate"
    if score < 0.50:  return "significant"
    return "major"


# --------------------------------------------------------------------------
# Mapper graph (KeplerMapper)
# --------------------------------------------------------------------------

def build_mapper_html(Xd: np.ndarray, labels: np.ndarray | None = None,
                      anomaly_scores: np.ndarray | None = None) -> str | None:
    """Build a KeplerMapper interactive HTML graph of the document corpus.

    Each node = a cluster of documents sharing similar vocabulary.
    Edge = clusters that share at least one document (overlap).
    Node colour = average anomaly score of documents inside it.
    Returns None if kmapper is not installed.
    """
    if not _KMAPPER:
        return None

    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import DBSCAN

    # Project to 2D for the lens (filter function)
    n_components = min(2, Xd.shape[1], Xd.shape[0] - 1)
    lens = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(Xd)
    lens = MinMaxScaler().fit_transform(lens)

    mapper = km.KeplerMapper(verbose=0)

    graph = mapper.map(
        lens,
        Xd,
        clusterer=DBSCAN(eps=0.5, min_samples=2, metric="cosine"),
        cover=km.Cover(n_cubes=10, perc_overlap=0.5),
    )

    # Colour nodes by mean anomaly score — shape (n_docs, 1)
    if anomaly_scores is not None:
        amax = anomaly_scores.max() or 1.0
        node_color_fn = (anomaly_scores / amax).reshape(-1, 1)
        color_function_name = ["anomaly score"]
    else:
        node_color_fn = None
        color_function_name = None

    html = mapper.visualize(
        graph,
        title="Corpus shape — document neighbourhood map",
        color_function=node_color_fn,
        color_function_name=color_function_name,
    )
    return html


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class TDAPatternEngine:
    def __init__(self, random_state: int = 7):
        self.random_state = random_state
        self.docs: list[Document] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.X = None
        self.terms: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.anomaly_scores: np.ndarray | None = None
        self.entities: dict[str, list[tuple[str, str]]] = {}
        self._tda: dict = {}
        self._graph_store = _GraphStore()

    # ---- ingest ----------------------------------------------------------
    def ingest(self, docs: list[Document]) -> "TDAPatternEngine":
        self.docs = docs
        return self

    _NAME_STOPWORDS = {
        "anna", "müller", "anna müller",
        "stefan", "braun", "stefan braun",
        "thomas", "fischer", "thomas fischer",
        "julia", "lange", "julia lange",
        "eva", "hoffmann", "eva hoffmann",
        "maria", "schmidt", "maria schmidt",
        "klaus", "weber", "klaus weber",
        "priya", "anand", "priya anand",
        "omar", "faris", "omar faris",
        "marcus", "vale", "marcus vale",
        "yuki", "sato", "yuki sato",
        "lena", "roth", "lena roth",
        "david", "okoye", "david okoye",
        "tomas", "berg", "tomas berg",
        "sara", "klein", "sara klein",
    }

    # ---- vectorize -------------------------------------------------------
    def vectorize(self) -> "TDAPatternEngine":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop = list(set(ENGLISH_STOP_WORDS) | self._NAME_STOPWORDS)
        self.vectorizer = TfidfVectorizer(
            stop_words=stop, max_df=0.6, min_df=2, ngram_range=(1, 2)
        )
        self.X = self.vectorizer.fit_transform(d.text for d in self.docs)
        self.terms = np.array(self.vectorizer.get_feature_names_out())
        return self

    # ---- themes: DBSCAN --------------------------------------------------
    def find_themes(self) -> "TDAPatternEngine":
        """DBSCAN: density-based, no k needed, naturally finds noise points."""
        Xd = self.X.toarray()
        # cosine distance proxy: 1 - cosine_sim; experiment with eps
        eps_values = [0.35, 0.45, 0.55, 0.65]
        best_labels, best_n = None, 0
        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=3, metric="cosine")
            labels = db.fit_predict(Xd)
            n = len(set(labels) - {-1})
            if n > best_n:
                best_labels, best_n = labels, n
        if best_labels is None or best_n == 0:
            best_labels = np.zeros(len(self.docs), dtype=int)
        self.labels = best_labels
        self._n_noise = int((best_labels == -1).sum())
        return self

    def _top_terms(self, doc_indices, n=6) -> list[str]:
        if len(doc_indices) == 0:
            return []
        centroid = np.asarray(self.X[doc_indices].mean(axis=0)).ravel()
        top = centroid.argsort()[::-1][:n]
        return [self.terms[i] for i in top if centroid[i] > 0]

    def _keyword_hits(self, i) -> list[str]:
        text = self.docs[i].text.lower()
        return [kw for kw in SUSPICIOUS_VOCAB if kw in text]

    def _distinctive_terms(self, i, n=6) -> list[str]:
        row = np.asarray(self.X[i].todense()).ravel()
        mean = np.asarray(self.X.mean(axis=0)).ravel()
        delta = row - mean
        top = delta.argsort()[::-1][:n]
        return [self.terms[j] for j in top if delta[j] > 0]

    # ---- anomalies: IsolationForest + Autoencoder + giotto-tda + keyword ----
    def find_anomalies(self, contamination: float = 0.08) -> "TDAPatternEngine":
        """Four signals:
          1. Isolation Forest — classical tree-based outlier detection
          2. Autoencoder reconstruction error — deep representation anomaly
          3. Keyword density — suspicion vocabulary
          4. Local topology entropy (giotto-tda) — per-doc neighbourhood complexity
        """
        Xd = self.X.toarray()

        # 1. Isolation Forest
        iso = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=200,
        )
        iso.fit(Xd)
        iso_raw = -iso.decision_function(Xd)

        # 2. Autoencoder reconstruction error
        ae_errors = _train_autoencoder(Xd)

        # 3. Keyword density
        kw_counts = np.array([len(self._keyword_hits(i)) for i in range(len(self.docs))], dtype=float)

        # 4. Per-document local topology entropy via giotto-tda
        topo_dev = _local_topo_scores(Xd)

        def norm(v):
            rng = np.ptp(v)
            return (v - v.min()) / (rng + 1e-9)

        # keyword stays at 0.30 — topo_dev gets 0.15 as the new 4th signal
        self.anomaly_scores = (
            0.30 * norm(iso_raw)
            + 0.25 * norm(ae_errors)
            + 0.30 * norm(kw_counts)
            + 0.15 * norm(topo_dev)
        )
        cutoff = np.quantile(self.anomaly_scores, 1 - contamination)
        self._anomaly_flag = self.anomaly_scores >= cutoff
        self._iso_scores = iso_raw
        self._ae_errors = ae_errors
        self._topo_dev = topo_dev
        return self

    # ---- topological analysis --------------------------------------------
    def find_topology(self) -> "TDAPatternEngine":
        Xd = self.X.toarray()
        self._tda = _tda_fingerprint(Xd)
        return self

    # ---- entities + relationships ----------------------------------------
    def extract_entities(self) -> "TDAPatternEngine":
        self.entities = {}
        for d in self.docs:
            found = []
            for etype, pat in ENTITY_PATTERNS.items():
                for m in pat.findall(d.text):
                    val = m.strip()
                    if etype == "actor" and val.split()[0] in {
                        "The", "This", "Our", "Per", "All", "No", "Any", "Each",
                        "Both", "Low", "High", "New", "Old", "Key", "Standard",
                        "Follow", "Quick", "Noted", "Added", "Same", "Strong",
                    }:
                        continue
                    found.append((etype, val))
            self.entities[d.id] = found
        return self

    def build_relationships(self, min_shared: int = 1) -> list[Finding]:
        ent_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
        for doc_id, ents in self.entities.items():
            for e in ents:
                ent_docs[e].add(doc_id)

        actors = {e: docs for e, docs in ent_docs.items() if e[0] == "actor" and len(docs) >= 2}
        rels: list[Finding] = []
        raw_rels: list[dict] = []
        actor_list = list(actors.items())
        for i in range(len(actor_list)):
            for j in range(i + 1, len(actor_list)):
                (e1, d1), (e2, d2) = actor_list[i], actor_list[j]
                shared = d1 & d2
                if len(shared) >= min_shared:
                    rels.append(Finding(
                        kind="relationship",
                        title=f"{e1[1]} ↔ {e2[1]}",
                        detail=f"Co-occur in {len(shared)} documents.",
                        score=float(len(shared)),
                        sources=sorted(shared),
                        extra={"a": e1[1], "b": e2[1], "weight": len(shared)},
                    ))
                    raw_rels.append({"a": e1[1], "b": e2[1], "weight": len(shared)})
        rels.sort(key=lambda f: f.score, reverse=True)
        self._graph_store.store(raw_rels)
        return rels

    # ---- assemble findings -----------------------------------------------
    def findings(self) -> list[Finding]:
        out: list[Finding] = []

        # DBSCAN themes (skip noise cluster -1)
        for c in sorted(set(self.labels) - {-1}):
            idx = np.where(self.labels == c)[0]
            terms = self._top_terms(idx)
            out.append(Finding(
                kind="theme",
                title="Cluster: " + ", ".join(terms[:3]),
                detail=f"{len(idx)} documents. DBSCAN cluster {c}. Key terms: " + ", ".join(terms),
                score=float(len(idx)),
                sources=[self.docs[i].id for i in idx],
                extra={"terms": terms, "size": int(len(idx)), "cluster_id": int(c)},
            ))

        # anomalies / suspicious
        flagged = np.where(self._anomaly_flag)[0]
        order = flagged[np.argsort(self.anomaly_scores[flagged])[::-1]]
        for i in order:
            d = self.docs[i]
            why = self._distinctive_terms(i)
            keywords = self._keyword_hits(i)
            kind = "suspicious" if keywords else "anomaly"
            title_prefix = "Suspicious" if keywords else "Anomaly"
            out.append(Finding(
                kind=kind,
                title=f"{title_prefix} in {d.id}",
                detail=d.text[:140] + ("…" if len(d.text) > 140 else ""),
                score=float(self.anomaly_scores[i]),
                sources=[d.id],
                extra={
                    "meta": d.meta,
                    "terms": why,
                    "keywords": keywords,
                    "iso_score": round(float(self._iso_scores[i]), 4),
                    "ae_error": round(float(self._ae_errors[i]), 6),
                    "topo_entropy": round(float(self._topo_dev[i]), 6),
                },
            ))

        out.extend(self.build_relationships())
        return out

    # ---- traceable query -------------------------------------------------
    def query(self, q: str, top: int = 5) -> list[dict[str, Any]]:
        qv = self.vectorizer.transform([q])
        sims = (self.X @ qv.T).toarray().ravel()
        order = sims.argsort()[::-1][:top]
        results = []
        for i in order:
            if sims[i] <= 0:
                continue
            d = self.docs[i]
            results.append({
                "doc_id": d.id,
                "score": round(float(sims[i]), 4),
                "snippet": d.text,
                "meta": d.meta,
                "theme": int(self.labels[i]) if self.labels is not None else None,
                "anomaly_score": round(float(self.anomaly_scores[i]), 4)
                if self.anomaly_scores is not None else None,
            })
        return results

    # ---- run everything --------------------------------------------------
    def run(self) -> dict[str, Any]:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.vectorize().find_themes().find_anomalies().find_topology().extract_entities()
        all_findings = self.findings()
        themes = [f for f in all_findings if f.kind == "theme"]
        return {
            "n_docs": len(self.docs),
            "n_clusters": len(themes),
            "n_noise_docs": self._n_noise,
            "graph_backend": self._graph_store.backend,
            "tda": self._tda,
            "torch_available": _TORCH,
            "ripser_available": _RIPSER,
            "giotto_available": _GIOTTO,
            "neo4j_available": self._graph_store.backend == "neo4j",
            "findings": [asdict(f) for f in all_findings],
        }
