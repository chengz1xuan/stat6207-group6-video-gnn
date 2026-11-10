# Mplifier Graph Probe — Hang Seng Bank 20

## Purpose

This is a controlled **graph probe / graph-readiness / clustering test** built from the Mplifier export `ad-reference-vault-export-2026-08-22 (1).json`.

It is **not** a claim that Mplifier already has a working GNN. The purpose is to give Ming enough structured advertising data to judge whether Mplifier's current decomposition is useful for clustering, graph intelligence and possible future GNN-style research without exposing internal prompts, private repository structure, provider secrets or the full proprietary workflow.

## Dataset diagnostic

- Exported references: **20**
- Included references: **20**
- Clearly off-theme exclusions: **0**
- Parent brand after normalization: **Hang Seng Bank**
- Official `恒生銀行 Hang Seng Bank` author/channel records: **20 / 20**
- Video nodes: **20**
- Aspect nodes: **794**
- Evidence nodes: **1244**
- Candidate edges: **4703**

All 20 records were retained. Canonical brand strings vary in capitalization and Chinese/English ordering — for example `Hang Seng Bank`, `HANG SENG BANK`, `恒生銀行 Hang Seng Bank`, and bilingual variants. The graph probe normalizes these to `brand_family = Hang Seng Bank` while preserving the original value in `brand_raw`.

All 20 source records are from the official `恒生銀行 Hang Seng Bank` YouTube channel, so `same_author` will be a saturated relationship in this dataset. That is useful as a reminder that deterministic edges can be valid but still low-information for clustering.

## Critical caveat about edges

**The edge table is not ground truth. It is a candidate relationship table for review.**

Treat edge classes differently:

1. **Structural / deterministic edges**
   - `video_has_aspect`
   - `video_has_evidence`
   - exact-match `aspect_supported_by_evidence`
   - `same_brand`
   - `same_author`
   - `same_creative_mechanic`
   - `same_structure_type`
   - `same_evidence_mode`
   - `same_cta_explicitness`
   - `same_pacing`
   - `same_safety_status`

2. **Rule-based / weaker evidence-support edges**
   - some `aspect_supported_by_evidence` rows with `deterministic=false`

3. **Similarity candidates**
   - `similar_aspect_candidate`
   - generated only between the same aspect family across different videos
   - character n-gram TF-IDF cosine similarity
   - candidate threshold: `>= 0.48`
   - **must not be treated as semantic truth or a training label without review**

## Files

### `video_nodes.csv`
One row per included reference. Preserves:
- brand / normalized brand family / heuristic sub-brand
- product/campaign
- title
- main message
- hook
- CTA
- value promise
- creative mechanic
- structure type
- evidence mode
- CTA explicitness
- proof strength
- pacing
- platform fit
- safety status
- copy risk
- fit score
- unresolved evidence counts
- `what_not_to_copy`

### `aspect_nodes.csv`
Atomic or semi-atomic creative, strategic, quality, reuse, audience and safety aspects. Includes source paths, confidence, graph role and similarity eligibility.

### `evidence_nodes.csv`
Atomic evidence split from source arrays. `evidence_class` distinguishes:
- `source_raw`
- `normalized_reading`
- `quality_flag`

### `candidate_edges.csv`
Candidate graph relationships with:
- edge type
- strength
- deterministic flag
- score
- construction basis
- review note

### `graph_probe_hang_seng_20.jsonl`
One video graph bundle per line with its video node, aspect nodes, evidence nodes and touching edge IDs.

### `graph_probe_hang_seng_20_summary.json`
Machine-readable diagnostics, distributions and edge counts.

### `message_to_ming.md`
Paste-ready communication.

## Brand and sub-brand normalization

Canonical brand strings are normalized to:
`brand_family = Hang Seng Bank`

The canonical brand remains in `brand_raw`.

For exploratory graph testing only, a heuristic `sub_brand` field is also generated from brand/product cues, e.g.:
- Hang Seng Credit Card
- Hang Seng Preferred Banking
- Hang Seng Prestige Banking
- Hang Seng Insurance
- Hang Seng Bank

This heuristic label is **not ground truth** and should not be used as a supervised target without review.

## Why this set is useful after McDonald's and HSBC

The earlier McDonald's pack tested a controlled same-brand set. The HSBC pack added stronger internal variation across a parent brand. This Hang Seng set is useful because all 20 records come from the **same official channel and same parent brand**, but span multiple products and campaign styles — including credit cards, +FUN Dollars, travel, banking tiers, offers and celebrity-led executions.

That makes it a clean test of whether creative-intelligence fields can help separate records when `brand` and `author` themselves provide almost no clustering discrimination.

In other words: if meaningful clusters appear here, they are more likely to be driven by **product, hook, value promise, CTA, structure, evidence mode and creative mechanic**, rather than simply brand or source identity.

## IP boundary

This package contains structured export data and graph-test transformations only. It does **not** include Mplifier internal prompts, private code/repository structure, provider secrets or the complete production workflow.
