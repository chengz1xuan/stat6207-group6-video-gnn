# Mplifier — Prudential graph-readiness dataset (15 clean cases)

Standalone Ming research package from `ad-reference-vault-export-Prudential.json`. No Mplifier MVP modifications.

## Uploader audit
- Selected: **15**
- Official uploader matches: **15**
- Third-party excluded: **0**
- Unresolved held out: **0**
- Final clean: **15**
- Expected official author: `Prudential Hong Kong Limited 香港保誠保險`
- Official channel identities (both same publisher): @prudentialhk, UC0FcaFJVwvfwIRgPJjFJaKQ
- Independent registry check: Prudential Hong Kong official website lists the Prudential Hong Kong Limited YouTube channel; YouTube channel UC0FcaFJVwvfwIRgPJjFJaKQ identifies handle @prudentialhk. Matching requires both this author and one known channel identity.
- Reference: https://www.prudential.com.hk/tc/claims-and-services/tools/customer-notices/security-alerts/

## Graph inventory
- Video nodes: **15**
- Aspect nodes: **659**
- Evidence nodes: **804**
- Candidate edges: **3073**

## Contract / guidance
`video_nodes.csv` follows Ming's exact ordered 35-column schema. Source `author` and reference IDs are unchanged. `brand_family` groups the parent brand; `sub_brand` remains conservatively equal to parent brand if no independently verified subbrand taxonomy exists. `product_campaign` retains product or campaign names.

`aspect_nodes.csv` contains source-/analysis-derived attribute atoms, `evidence_nodes.csv` preserves raw and normalized evidence atoms, and `candidate_edges.csv` mixes deterministic membership/same-field links with explicitly nondeterministic text similarity and evidence-support candidates. **No candidate edge is a learned GNN label.**

## Package files
1. README_data_dictionary.md
2. video_nodes.csv
3. aspect_nodes.csv
4. evidence_nodes.csv
5. candidate_edges.csv
6. graph_probe_prudential_15.jsonl
7. graph_probe_prudential_15_summary.json
8. message_to_ming.md
