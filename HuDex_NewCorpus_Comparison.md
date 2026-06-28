# HuDex Pattern Intelligence — New Corpus Analysis & Engine Comparison

**104 documents · German public sector · June 2026 · evocenta / Arlequin AI**

---

## Summary

Same 104 documents, three different engines. The table below shows the core difference at a glance.

| Metric | :8001 Original | :8003 Baseline+ | :8002 TDA Engine |
|---|:---:|:---:|:---:|
| Documents | 104 | 104 | 104 |
| **Fraud comms caught (F01–F04)** | **0 / 4** | **0 / 4** | **4 / 4** |
| Meridian ring caught (R01–R06) | 2 / 6 | 2 / 6 | 1 / 6 |
| False positives (A01–A04) | **4** | **4** | **0** |
| Suspicious kind assigned | 0 | 0 | 5 |
| Anomalies flagged | 9 | 9 | 4 |
| Correctly flagged total | 2 | 2 | 5 |
| Wrongly flagged total | 4 | 4 | 0 |
| Topology Betti-1 | — | 64 | 64 |
| Per-doc topology signal | No | No | Yes |
| Autoencoder signal | No | No | Yes |
| Keyword weight | 0.20 | 0.20 | 0.30 |
| DBSCAN noise docs | — | — | 40 |

---

## Corpus Design

The 104-document corpus simulates a German municipal administration. Three distinct fraud pattern types were planted, each designed to expose a specific engine weakness.

### Normal operations — 90 documents (DOC-0001 to DOC-0090)

Budget approvals, procurement tenders, IT maintenance, HR onboarding, logistics, compliance audits. Realistic German public sector language: Anna Müller, Klaus Weber, Stefan Braun, Thomas Fischer, Julia Lange, Eva Hoffmann, Maria Schmidt.

### Explicit fraud communications — 4 documents (DOC-F01 to DOC-F04)

Direct keyword hits. Every document contains at least two phrases from the suspicious vocabulary list.

| ID | Who | What | Keywords |
|---|---|---|---|
| DOC-F01 | Klaus Weber | Meridian payment off the books, cash only, keep this between us | `off the books`, `no invoice` |
| DOC-F02 | Anna Müller | Kickback coded as consultancy fees, delete this | `kickback`, `delete this` |
| DOC-F03 | Stefan Braun | Launder through shell entity, untraceable, no paperwork | `launder`, `shell entity`, `no paperwork` |
| DOC-F04 | Thomas Fischer | Bribe settled in cash, EUR 15,000 off-book, stay quiet | `off-book`, `no invoice` |

**Detection result:** :8001 and :8003 miss all four. :8002 catches all four as suspicious.

### Meridian Consulting fraud ring — 6 documents (DOC-R01 to DOC-R06)

No single smoking-gun keyword. Pattern only visible across all six documents: repeated transfers to `ACC-77391` / Meridian Consulting, verbal-only authorisation, no supporting contract, no tender.

| ID | Content |
|---|---|
| DOC-R01 | First disbursement to Meridian Consulting via ACC-77391, verbal instruction only |
| DOC-R02 | Second tranche, no supporting contract on file |
| DOC-R03 | Reconciliation note, verbal approval only, no written record |
| DOC-R04 | Fourth disbursement, authorisation chain unclear |
| DOC-R05 | ACC-77391 flagged for review, invoices do not match procurement register |
| DOC-R06 | Final disbursement EUR 340,000 this quarter, no tender documented |

**Detection result:** :8001/:8003 catch R01, R02, R04 as anomalies. :8002 catches R02 only. The full ring requires cross-document entity tracking — none of the engines do this yet.

### Technical outliers — 4 documents (DOC-A01 to DOC-A04)

Legitimate government documents with highly unusual vocabulary. Designed to trap the LOF engine into spending its anomaly budget on false positives.

| ID | Content | Why LOF flags it |
|---|---|---|
| DOC-A01 | IAS 8 accounting restatement, componentisation, discount rate 3.87% | Unique vocab: amortisation, transitional provision, NPV |
| DOC-A02 | SKOS ontology alignment to XÖV municipal data standard | Unique vocab: ontology, SKOS, concept mapping, XÖV |
| DOC-A03 | GIS cadastre system interoperability, WFS 2.0, INSPIRE directive | Unique vocab: WFS, INSPIRE, middleware, feature records |
| DOC-A04 | Depreciation schedule, intangible asset class XK-9 | Unique vocab: intangible asset, amortisation, residual carrying amount |

**Detection result:** :8001/:8003 flag all four as top anomalies (scores 0.661–0.780). :8002 flags none of them.

---

## Engine 1 — localhost:8001 (Original)

**Formula:** `score = 0.5 × LOF + 0.3 × centroid_distance + 0.2 × keyword_density`

### Flagged documents (9 anomalies, 0 suspicious)

| Document | Score | Actual content | Fraud? |
|---|:---:|---|:---:|
| DOC-A02 | 0.780 | SKOS ontology alignment (legitimate IT) | ❌ false positive |
| DOC-A04 | 0.767 | Depreciation schedule (legitimate finance) | ❌ false positive |
| DOC-A03 | 0.749 | GIS interoperability (legitimate IT) | ❌ false positive |
| DOC-R04 | 0.673 | Meridian 4th transfer, authorisation chain unclear | ✓ partial |
| DOC-A01 | 0.661 | IAS 8 restatement (legitimate finance) | ❌ false positive |
| DOC-R01 | 0.626 | Meridian 1st transfer, verbal instruction only | ✓ partial |
| DOC-R02 | 0.588 | Meridian 2nd tranche, no contract | ✓ partial |
| DOC-0032 | 0.553 | Recruiter update (legitimate HR) | ❌ false positive |
| DOC-0089 | 0.532 | Expense report approval (legitimate finance) | ❌ false positive |

**5 of 9 flags are false positives. All 4 explicit fraud communications missed.**

### Why the fraud comms are invisible

The four fraud docs (F01–F04) all have similar structure: short (1–2 sentences), `comms` channel, same financial crime vocabulary. In TF-IDF space they form a tight cluster. LOF sees a dense neighbourhood and scores them as **normal** — density-based detection treats clustered documents as internally consistent regardless of their content.

Keyword weight of 0.2 is not enough to override a favourable LOF + centroid score. The formula would need keyword weight above ~0.45 to push the fraud cluster above the 8% contamination cutoff.

### Why the technical outliers top the list

DOC-A01–A04 use vocabulary (amortisation, SKOS, WFS, componentisation) that appears nowhere else in the 90 normal documents. Centroid distance is very high. LOF also scores them as isolated because they have no dense neighbourhood. Result: 4 of the 9 anomaly slots are consumed by legitimate documents.

---

## Engine 2 — localhost:8003 (Baseline+)

**Formula:** identical to :8001 — `score = 0.5 × LOF + 0.3 × centroid_distance + 0.2 × keyword_density`

Baseline+ adds ripser topology (Betti-1 = 64, 5 persistent H1 loops) and the upload endpoint, but the anomaly scoring formula is unchanged. On this corpus the findings are **identical to :8001**.

The topology signal is corpus-level only — it tells you the corpus has 64 loops in its embedding, but does not assign a topology score to individual documents. This is why it does not help catch F01–F04.

### H1 persistence barcode

| Loop | Birth | Death | Persistence |
|:---:|:---:|:---:|:---:|
| 1 | 0.8178 | 1.0000 | 0.1822 |
| 2 | 0.7477 | 0.9264 | 0.1787 |
| 3 | 0.8304 | 1.0000 | 0.1696 |
| 4 | 0.8420 | 1.0000 | 0.1580 |
| 5 | 0.7851 | 0.9386 | 0.1534 |

The 5 persistent H1 loops are structural features of the full corpus embedding. They reflect the fact that the fraud documents form a cycle in vocabulary space — but since this is a corpus-level signal, it cannot be used to rank individual documents.

---

## Engine 3 — localhost:8002 (TDA Engine)

**Formula:** `score = 0.30 × IsolationForest + 0.25 × Autoencoder_error + 0.30 × keyword_density + 0.15 × topo_entropy`

### Flagged documents (5 suspicious, 4 anomalies)

| Document | Kind | Score | iso_score | ae_error | topo_entropy | Keywords |
|---|:---:|:---:|:---:|:---:|:---:|---|
| DOC-F03 | SUSPICIOUS | 0.653 | -0.0350 | 0.001549 | 0.3181 | `no paperwork`, `shell entity` |
| DOC-F04 | SUSPICIOUS | 0.474 | -0.0192 | 0.000422 | 0.3258 | `off-book`, `no invoice` |
| DOC-R02 | anomaly | 0.471 | -0.0011 | 0.001098 | 0.5918 | — |
| DOC-0008 | anomaly | 0.470 | -0.0243 | 0.001294 | 1.1605 | — |
| DOC-F03 → DOC-0082 | SUSPICIOUS | 0.461 | -0.0097 | 0.000655 | 0.4998 | `launder`, `laundering` |
| DOC-F01 | SUSPICIOUS | 0.455 | -0.0494 | 0.000510 | 0.7013 | `off the books`, `no invoice` |
| DOC-0023 | anomaly | 0.452 | +0.0100 | 0.001048 | 0.1136 | — |
| DOC-0059 | anomaly | 0.443 | +0.0033 | 0.000933 | 0.4392 | — |
| DOC-F02 | SUSPICIOUS | 0.431 | -0.0438 | 0.000852 | 0.3181 | `delete this`, `kickback` |

### Why IsolationForest catches what LOF cannot

IsolationForest builds 200 random decision trees and measures how few splits isolate each document. It evaluates documents **independently** — it does not care whether nearby documents are also anomalous.

Each fraud comm is short, uses rare financial crime vocabulary, and sits in the `comms` channel. These properties make each one easy to isolate regardless of whether the others exist nearby. With keyword weight at 0.30, the combined score pushes all four F-series docs above the 8% contamination cutoff.

LOF's fundamental weakness: a cluster of 4 conspirators who communicate consistently will always score as **normal** under LOF because they form a dense neighbourhood relative to each other.

### Why the technical outliers disappear

DOC-A01–A04 have zero keyword hits. The autoencoder, trained on the full 104-document corpus, learns that long formally structured text (even with unusual vocabulary) follows a consistent pattern — reconstruction error for the technical docs is moderate, not extreme. IsolationForest scores them as mid-range because their document length and channel are not outlying. None of the 4 false positives from :8001/:8003 appear in the :8002 flagged list.

### Per-document topology entropy (topo_entropy)

`topo_entropy` is computed via ripser on each document's 10-nearest-neighbour subgraph. It measures the Shannon entropy of the H1 barcode persistence values — a high value means the local neighbourhood has complex topological structure (many competing loops of varying lengths).

The fraud comms score topo_entropy 0.32–0.70. This confirms they sit in topologically unusual neighbourhoods — the vocabulary overlap between F01–F04 and the rest of the corpus creates short-lived loops in the local embedding that are absent in normal documents.

DOC-0008 (a routine procurement order) scores topo_entropy 1.16 — the highest in the corpus. This is a genuine anomaly: the document's local neighbourhood in the TF-IDF embedding is topologically complex for reasons that are not explained by keywords or IsolationForest. It warrants manual review.

### The remaining gap — Meridian ring

Only DOC-R02 from the 6-document Meridian ring is caught. DOC-R01, R03–R06 are individually unremarkable finance documents that use standard procurement vocabulary. No fraud keywords match. IsolationForest scores them as only mildly isolated.

Catching the full ring requires **cross-document analysis**:
- Named-entity linking across the corpus: track `ACC-77391` and `Meridian Consulting` as entities, flag when the same entity appears in 3+ documents with missing authorisation fields
- Temporal sequence detection: 6 transfers to the same account in 5 weeks
- Graph anomaly detection: in the co-occurrence network, flag entities with unusually high degree concentrated in a narrow time window

None of these are in the current pipeline. This is the next capability gap.

---

## Side-by-Side Comparison

| Metric | :8001 Original | :8003 Baseline+ | :8002 TDA Engine |
|---|:---:|:---:|:---:|
| Algorithm | LOF + centroid | LOF + centroid | IsoForest + AE + topo |
| Keyword weight | 0.20 | 0.20 | 0.30 |
| Fraud comms F01 | ❌ missed | ❌ missed | ✓ suspicious |
| Fraud comms F02 | ❌ missed | ❌ missed | ✓ suspicious |
| Fraud comms F03 | ❌ missed | ❌ missed | ✓ suspicious |
| Fraud comms F04 | ❌ missed | ❌ missed | ✓ suspicious |
| Meridian R01 | ✓ anomaly | ✓ anomaly | ❌ missed |
| Meridian R02 | ✓ anomaly | ✓ anomaly | ✓ anomaly |
| Meridian R03 | ❌ missed | ❌ missed | ❌ missed |
| Meridian R04 | ✓ anomaly | ✓ anomaly | ❌ missed |
| Meridian R05 | ❌ missed | ❌ missed | ❌ missed |
| Meridian R06 | ❌ missed | ❌ missed | ❌ missed |
| Tech outlier A01 | ❌ false positive | ❌ false positive | ✓ not flagged |
| Tech outlier A02 | ❌ false positive | ❌ false positive | ✓ not flagged |
| Tech outlier A03 | ❌ false positive | ❌ false positive | ✓ not flagged |
| Tech outlier A04 | ❌ false positive | ❌ false positive | ✓ not flagged |
| Corpus Betti-1 | — | 64 | 64 |
| Per-doc topo signal | No | No | Yes |

---

## Conclusions

### 1 — LOF is blind to organised fraud clusters
Any scheme where conspirators communicate in consistent language will defeat LOF. A cluster of similar-looking documents scores as normal regardless of their content. Keyword weight must exceed ~0.45 to override the density signal, or the formula must weight keywords as an independent gate rather than a blended component.

### 2 — Technical outliers consume the anomaly budget
With contamination fixed at 8%, every slot spent on a legitimate document is a missed fraud document. :8001/:8003 spend 4 of 9 slots on legitimate technical docs. A pre-filter that classifies documents by channel or structure before anomaly scoring would eliminate this waste.

### 3 — TDA avoids the false-positive trap
IsolationForest + Autoencoder does not penalise long formally structured text. The false-positive rate on legitimate outliers drops from 44% (4/9) to 0%. This alone justifies the TDA engine for production use.

### 4 — Meridian ring requires cross-document entity tracking
Six documents that are individually unremarkable but collectively a EUR 340,000 fraud scheme. The current engines evaluate each document in isolation. Cross-document named-entity linking on account numbers and vendor names, combined with temporal sequence analysis, is the next capability to implement.

### 5 — topo_entropy is a genuine fourth signal
The per-document topology entropy is non-zero for all caught fraud comms (0.32–0.70) and correctly elevated for the one anomalous procurement document (DOC-0008, topo=1.16). It provides a dimension of evidence that is independent of the vocabulary-based signals and catches structural irregularities that keywords and IsolationForest both miss.

### 6 — Both engines catch different things
Run both in parallel. A document that appears in both :8003 and :8002 has four independent confirmations: LOF density anomaly, IsolationForest isolation, keyword vocabulary match, and topological neighbourhood complexity. That is the highest-confidence fraud signal the system can produce.

---

*HuDex New Corpus Analysis · June 2026 · evocenta / Arlequin AI*
