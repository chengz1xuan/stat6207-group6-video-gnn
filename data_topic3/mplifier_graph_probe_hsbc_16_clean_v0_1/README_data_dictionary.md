# Mplifier Graph Probe — HSBC 16

## Purpose
Controlled **graph probe / graph-readiness / clustering test** from `ad-reference-vault-export-2026-08-22.json`. This is **not** a claim that Mplifier already has a working GNN.

The pack exposes structured exported ad data and graph-test transformations only. It does not expose internal prompts, private repository structure, provider secrets, or the full production workflow.

## Diagnostic
- Exported references: **16**
- Included references: **16**
- Clearly off-theme exclusions: **0**
- Parent brand after normalization: **HSBC**
- `HSBC Hong Kong` author/channel: **15**
- Third-party archive author/channel: **1**
- Video nodes: **16**
- Aspect nodes: **704**
- Evidence nodes: **1037**
- Candidate edges: **3554**

All 16 were retained as HSBC-family material. Several UI-level brand fields are noisy/product-like, so identity is based on canonical brand/evidence. `HSBC Life` is preserved as a sub-brand while grouped under the HSBC parent family.

The third-party source is a 1995 HSBC commercial from a Hong Kong TV-ad archive. The source author differs, but the advertised brand is clearly HSBC, so it remains in the set.

## Critical edge caveat
**The edge table is not ground truth. It is a candidate relationship table for review.**

Deterministic/structural:
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

Weaker:
- non-exact `aspect_supported_by_evidence`
- `similar_aspect_candidate`

`similar_aspect_candidate` uses same-aspect-family character n-gram TF-IDF cosine similarity with a candidate threshold of **0.48**. It is a review proposal, not a semantic label.

Important example: all 16 records currently have `medium tempo`. Therefore `same_pacing` is deterministic but has very little discriminative value in this dataset.

## Files and schemas

### `video_nodes.csv`
One row per reference. Preserves brand, sub-brand, product/campaign, title, main message, hook, CTA, value promise, creative mechanic, structure, evidence mode, CTA explicitness, proof strength, pacing, platform fit, safety, copy risk, fit score, unresolved evidence and what-not-to-copy guidance.

### `aspect_nodes.csv`
Atomic/semi-atomic creative, strategic, evidence, reuse, audience and safety aspects. Includes source path, source/derived class, confidence, graph role and similarity eligibility.

### `evidence_nodes.csv`
Atomic evidence split from raw and normalized export arrays. `evidence_class` distinguishes:
- `source_raw`
- `normalized_reading`
- `quality_flag`

### `candidate_edges.csv`
Columns:
- `edge_id`
- `source_node_id`
- `target_node_id`
- `edge_type`
- `edge_strength`
- `deterministic`
- `score`
- `basis`
- `review_note`
- `directed`

### `graph_probe_hsbc_16.jsonl`
One video bundle per line with its video node, aspects, evidence and touching edge IDs.

### `graph_probe_hsbc_16_summary.json`
Machine-readable counts, distributions, diagnostics and previous-pack comparison.

### `message_to_ming.md`
Paste-ready Ming communication.

## Brand normalization
For this probe:
- HSBC
- HSBC 滙豐
- 滙豐 (HSBC)
- HSBC Life

map to `brand_family = HSBC`.

The more specific canonical brand remains in `brand_raw`; `HSBC Life` remains identifiable in `sub_brand`.

This is a graph-test normalization decision, not a claim that all HSBC product lines are interchangeable.

## Versus previous McDonald's 20-case pack
Previous supplied baseline:
- 20 references
- 667 aspect nodes
- 1,486 evidence nodes
- 6,869 candidate edges

Current HSBC pack:
- 16 references
- 704 aspect nodes
- 1037 evidence nodes
- 3554 candidate edges

The more important difference is research shape: this is one parent brand with stronger internal heterogeneity across app/service utility, wealth/Premier, HSBC Life/insurance and brand-level work. That is useful for testing whether meaningful **sub-clusters** emerge inside one parent brand.

Counts are construction outputs, not model-performance metrics. Because the previous graph-builder implementation itself is not included in this package, node/edge count changes should not be treated as a strict apples-to-apples quality score.

## Suggested questions for Ming
- Do HSBC Life cases form a useful sub-cluster without manually forcing the label?
- Do app/service utility ads separate from wealth/Premier and insurance?
- Which aspect families carry useful cluster signal: hook, value promise, proof, CTA, mechanic, audience?
- Which deterministic edges are saturated/noisy and should be dropped?
- Is `similar_aspect_candidate` useful before graph embeddings or future GNN tests?
- What minimum data volume and reviewed labels would be needed before a GNN-style experiment?
