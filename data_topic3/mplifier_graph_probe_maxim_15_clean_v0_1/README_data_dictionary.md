# Mplifier / Ming — Maxim's independent graph-probe package

Input: `ad-reference-vault-export-Maxim.json` — 15 selected videos. **No master dataset modification.**

## Upload control
- Retained as official-brand uploads: 15 / 15
- Excluded third-party uploads: 0
- Unresolved uploaders held out: 0
- Original uploader: 美心食品 Maxim’s Products
- Accepted source identities: `UCbv_6SSiwvl7fLoXg4usdWA`, `@MaximsProducts` (one YouTube channel)
- Channel identity: https://www.youtube.com/channel/UCbv_6SSiwvl7fLoXg4usdWA
- Per-record audit with original author/channel and inclusion reason: `graph_probe_maxim_15_summary.json`.

## Graph nodes and edges
- Video nodes: 15
- Aspect nodes: 611
- Evidence nodes: 818
- Candidate edges: 2871

`video_nodes.csv` contains exactly Ming's 35 columns in the established order. `brand_family` is normalized to Maxim's, while original `brand_raw`, `top_level_ui_brand`, and `author` are preserved. Product-line `sub_brand` is an exploratory normalization, not a new source claim.

`aspect_nodes.csv`: one row per creative identity, interpretation, signal, caution or component. Includes type, value, source path and similarity eligibility.

`evidence_nodes.csv`: each record's raw metadata/transcript/extract atoms and selected normalized cues with source paths; content is not independently re-assessed.

`candidate_edges.csv`: structural membership, exact shared-field matches and char n-gram similarity suggestions. **None establishes causal or semantic ground truth; graph-ready probe only, not a trained GNN.**

`graph_probe_maxim_15.jsonl`: video bundles for graph probing. `graph_probe_maxim_15_summary.json`: audit, distributions, diagnostics. `message_to_ming.md`: message draft.
