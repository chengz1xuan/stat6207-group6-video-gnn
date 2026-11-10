# Mplifier — IKEA Hong Kong graph-readiness dataset (15 cases)

Controlled research pack for Ming. Input: `ad-reference-vault-export-Ikea.json`. No Mplifier MVP changes.

## Uploader audit
- Selected: 15
- Official retained: 15
- Third-party excluded: 0
- Unresolved: 0
- Registry: `IKEAHongKong`, channel ID `UCcQ5WlinXufbKizIJHlI4LQ` or handle `@IKEAHongKong`
- Authors are unmodified source metadata. The channel handle used in three references is recognized as an alias.
- No duplicate reference IDs or YouTube video URLs. Similar titles may represent different video edits/IDs.

## Outputs
- `video_nodes.csv`: 15 videos; Ming's 35 fixed headers/order; provenance, analysis and inclusion decision.
- `aspect_nodes.csv`: 611 extracted/derived creative attributes.
- `evidence_nodes.csv`: 744 source and normalized evidence atoms.
- `candidate_edges.csv`: 2926 graph candidate links.
- `graph_probe_ikea_15.jsonl`: per-video graph bundles.
- `graph_probe_ikea_15_summary.json`: audit and counts.
- `message_to_ming.md`: suggested delivery note.

## Graph interpretation
Deterministic membership and equality links differ from `similar_aspect_candidate` heuristic relationships. Non-exact candidate evidence-support links are explicitly marked `deterministic=false` and require review. This is a graph probe, **not GNN training ground truth**.
