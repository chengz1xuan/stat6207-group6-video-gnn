# Mplifier — Mannings graph-readiness dataset (15 clean cases)

Standalone Ming research package from `ad-reference-vault-export-Mannings.json`. No Mplifier MVP modifications.

## Uploader audit
- Selected: **15**
- Official uploader matches: **15**
- Third-party excluded: **0**
- Unresolved held out: **0**
- Final clean: **15**
- Expected official author: `萬寧Mannings`
- Official channel identities (both same publisher): @manningshongkong, UCtOHIOjPlO12F7XghNsJzCg
- Independent registry check: YouTube channel UCtOHIOjPlO12F7XghNsJzCg advertises itself as 萬寧Mannings and identifies its handle as @manningshongkong. Matching requires both this author and one known channel identity.
- Reference: https://www.youtube.com/channel/UCtOHIOjPlO12F7XghNsJzCg

## Graph inventory
- Video nodes: **15**
- Aspect nodes: **657**
- Evidence nodes: **801**
- Candidate edges: **2953**

## Contract / guidance
`video_nodes.csv` follows Ming's exact ordered 35-column schema. Source `author` and reference IDs are unchanged. `brand_family` groups the parent brand; `sub_brand` remains conservatively equal to parent brand if no independently verified subbrand taxonomy exists. `product_campaign` retains product or campaign names.

`aspect_nodes.csv` contains source-/analysis-derived attribute atoms, `evidence_nodes.csv` preserves raw and normalized evidence atoms, and `candidate_edges.csv` mixes deterministic membership/same-field links with explicitly nondeterministic text similarity and evidence-support candidates. **No candidate edge is a learned GNN label.**

## Package files
1. README_data_dictionary.md
2. video_nodes.csv
3. aspect_nodes.csv
4. evidence_nodes.csv
5. candidate_edges.csv
6. graph_probe_mannings_15.jsonl
7. graph_probe_mannings_15_summary.json
8. message_to_ming.md
