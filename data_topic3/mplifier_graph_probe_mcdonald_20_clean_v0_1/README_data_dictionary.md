# Mplifier Graph Probe — McDonald's same-brand subset v0.1 CLEAN

Generated from `ad-reference-vault-export-2026-07-25.json`.

## Why this pack exists

Ming said the current issue is likely data size and unfinished feature design. He also suggested using videos from the same brand. This pack is a **controlled same-brand McDonald's subset**.

## Cleaning note

The uploaded file had 21 references, but one item was not McDonald's:

- `[香港廣告](2026)大家樂 靚湯靚餸晚市套餐` — canonical brand: Café de Coral

That off-brand item was removed from this clean pack.

## Counts

- Video nodes: 20
- Aspect nodes: 667
- Evidence nodes: 1486
- Candidate edges: 6869

## Files

- `video_nodes.csv` — one row per McDonald's video/ad case.
- `aspect_nodes.csv` — extracted creative aspects: hook, structure, proof, CTA, value promise, product/campaign, evidence mode, creative mechanic, language, safety/copy-risk, visual signal, audience signal.
- `evidence_nodes.csv` — source evidence: spoken dialogue, on-screen text, end-card text, metadata, CTA, audience takeaway, unresolved evidence.
- `candidate_edges.csv` — first-pass graph relationships: video-has-aspect, video-has-evidence, aspect-supported-by-evidence, same-brand, same-author, same-mechanic, same-structure, same-evidence-mode, similar-aspect candidate.
- `graph_probe_mcdonald_20.jsonl` — one JSON object per video containing nested video/aspect/evidence data.
- `graph_probe_mcdonald_20_summary.json` — count and diagnostics summary.
- `message_to_ming.md` — paste-ready WhatsApp / email message.

## Important caveat

The edge table is **not ground truth**. It is a candidate relationship table for review.

For model training, treat deterministic same-brand/same-mechanic/same-evidence-mode edges differently from weak `similar_aspect_candidate` edges.
