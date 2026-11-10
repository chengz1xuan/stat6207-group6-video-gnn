# Mplifier Graph Probe — Lee Kum Kee / 15 clean cases

## Purpose
Controlled graph-readiness / clustering dataset for Ming. This is not a production GNN implementation.

## Uploader audit
- Selected references: **15**
- Official brand uploads: **15**
- Third-party uploads excluded: **0**
- Unresolved uploader matches: **0**
- Final clean references: **15**

Verification basis:
Official-uploader registry matched author 'leekumkeekitchen' and channel ID UCuVp1SVmuRD0-ADZ7AFdhyw. Lee Kum Kee's official web properties link to its YouTube content, and the channel describes Lee Kum Kee's brand/cooking mission.

## Graph counts
- Video nodes: **15**
- Aspect nodes: **602**
- Evidence nodes: **925**
- Candidate edges: **3034**

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
6. graph_probe_lee_kum_kee_15.jsonl
7. graph_probe_lee_kum_kee_15_summary.json
8. message_to_ming.md
