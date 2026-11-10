# Mplifier / Ming — Kee Wah Bakery (奇華餅家) independent graph-probe package

Input: `ad-reference-vault-export-Kee Wah.json` — 15 selected references. This is an **independent package; the cumulative master is unchanged**.

## Verified uploader control
- Official-brand channel: https://www.youtube.com/channel/UCNRqFSTPHIzVwrVnZAMj_2A
- Canonical YouTube handle: https://www.youtube.com/@keewahbakery
- Author: Kee Wah Bakery 奇華餅家
- Accepted identities: @keewahbakery, UCNRqFSTPHIzVwrVnZAMj_2A
- Retained: **15/15** official uploads
- Excluded: **0** other uploaders
- Unresolved: **0**

Any exclusions would be recorded with original author, channel and source URL in `graph_probe_kee_wah_15_summary.json`.

## File dictionary
1. `video_nodes.csv` — 15 clean videos; EXACT Ming 35-column schema and order; original author and source brand preserved.
2. `aspect_nodes.csv` — 635 creative identity, mechanics, component, strategy and caveat nodes, with source paths.
3. `evidence_nodes.csv` — 701 source metadata, extracts and normalized evidence/quality cues.
4. `candidate_edges.csv` — 2748 memberships, same-field links and heuristic similarity candidates, marked as deterministic or not.
5. `graph_probe_kee_wah_15.jsonl` — bundled video graph records.
6. `graph_probe_kee_wah_15_summary.json` — audit, distributions, diagnostics and limitations.
7. `README_data_dictionary.md` — this document.
8. `message_to_ming.md` — handoff note for researcher.

## Exactly 35 columns
`video_id`, `source_reference_id`, `title`, `brand_raw`, `brand_family`, `sub_brand`, `top_level_ui_brand`, `product_campaign`, `author`, `platform`, `source_url`, `publish_date`, `duration_seconds`, `main_message`, `hook`, `cta`, `value_promise`, `creative_mechanic`, `structure_type`, `evidence_mode`, `cta_explicitness`, `proof_strength`, `pacing`, `production_complexity`, `platform_fit`, `safety_status`, `copy_risk`, `fit_score`, `overall_evidence`, `unresolved_count`, `warning_count`, `languages`, `what_not_to_copy`, `inclusion_status`, `inclusion_reason`

## Research limitations
- This is a **graph-readiness/clustering probe**, not a trained GNN or supervised labels.
- Source-produced creative interpretations are carried over, not independently judged.
- `brand_family` normalizes qualifying videos to **Kee Wah Bakery**. `sub_brand` is the parent family in this controlled batch; product/campaign details are preserved separately, and source brand values are unchanged.
- Third-party records are excluded rather than repaired/fabricated.
