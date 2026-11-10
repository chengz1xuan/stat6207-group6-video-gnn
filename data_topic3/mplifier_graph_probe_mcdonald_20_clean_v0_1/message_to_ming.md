Ming, I also prepared a clean McDonald’s same-brand subset.

This follows your suggestion that the next data should include more videos from the same brand, because it gives a more controlled comparison than the mixed 124-case dataset.

The uploaded set had 21 references, but one was actually Café de Coral, so I removed that. This clean pack has 20 McDonald’s cases.

Files included:

1) video_nodes.csv — one row per McDonald’s/McDonaldsHK video
2) aspect_nodes.csv — hook, structure, proof, CTA, value promise, campaign/product, evidence mode, language, safety/copy-risk, visual signal
3) evidence_nodes.csv — spoken dialogue, on-screen text, end-card text, metadata, CTA, audience takeaway
4) candidate_edges.csv — video-has-aspect, video-has-evidence, aspect-supported-by-evidence, same-brand, same-author, same-mechanic, same-structure, same-evidence-mode, similar-aspect candidates
5) graph_probe_mcdonald_20.jsonl — machine-readable nested version

Important: the edges are candidate relationships, not confirmed truth labels yet.

My thinking:
- The 124-case pack is for broad schema testing.
- The McDonald’s-only pack is for controlled same-brand feature testing.
- If this direction works, I can collect more same-brand videos later, ideally 50–100 per major brand.

Could you review whether this McDonald’s subset helps identify which features are usable for clustering/GNN training/testing?
