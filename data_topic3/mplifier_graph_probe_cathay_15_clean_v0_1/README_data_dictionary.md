# Mplifier Graph Probe — Cathay / 15 clean cases

## Purpose
Controlled graph-readiness / clustering dataset for Ming. This is not a production GNN implementation.

## Uploader audit
- Selected references: **15**
- Official brand uploads: **15**
- Third-party uploads excluded: **0**
- Unresolved uploader matches: **0**
- Final clean references: **15**

Verification basis:
Official-uploader registry matched author 'cathaypacific' and channel ID UCDF__e4OyxESzW1snu1i0Bg. Cathay Pacific's official website links its YouTube presence to youtube.com/cathaypacific.

## Graph counts
- Video nodes: **15**
- Aspect nodes: **619**
- Evidence nodes: **846**
- Candidate edges: **2968**

## Ming `video_nodes.csv` contract
The CSV uses Ming's exact requested header order. The `author` value is preserved from source truth and is never rewritten.

## Edge caveat
`candidate_edges.csv` is not ground truth. Deterministic structural/same-field relations are separate from weaker `similar_aspect_candidate` relationships.

## Package files
1. README_data_dictionary.md
2. video_nodes.csv
3. aspect_nodes.csv
4. evidence_nodes.csv
5. candidate_edges.csv
6. graph_probe_cathay_15.jsonl
7. graph_probe_cathay_15_summary.json
8. message_to_ming.md
