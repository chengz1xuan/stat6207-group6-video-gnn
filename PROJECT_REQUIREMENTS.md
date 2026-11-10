# Final Project execution and submission instructions (updated TA guidance)

This document records the user's supplied course/TA requirements and the corresponding project design. It is not a substitute for the final measured report.

## Required experiment

1. Use the latest supplied package per brand, uniformly documented. Start from real YouTube total views, target `ln(1 + views)`. Do not use `fit_score` as a response label or infer absent engagement counts from creative characteristics.
2. Refresh supplied URLs from public YouTube metadata, recording exact count, source URL, retrieval timestamp, extraction field, response digest, channel ID/name, and failure/missingness. Retain archived source labels separately. Likes/comments are descriptive supplementary outcomes only where publicly available.
3. Use one approximately 80/20 campaign-group-disjoint train/test split. Rounded video counts should be as close to 80/20 as whole groups allow. Tune encoders, hyperparameters, edge rules and stopping only within training. Refit each final model on the full outer 80% after internal tuning.
4. Primary representation: **one video is a subgraph** with video/context root, response-free creative concepts, and selected textual evidence. Begin from actual supplied `candidate_edges` membership/support links. Exclude engagement/assessment nodes; flag heuristic support links.
5. Compare **structural-only** versus **structural + custom similarity edges**. Custom similarity is a tested heuristic; use training-fitted text transforms and fixed rule parameters. Include a no-message-passing control with matched node information.
6. Extension representation: **one video is a node**. Compare provided structural video relations and structural + feature-similarity links. Held-out videos must attach only to training references; no test-test message passing or test-label graph construction.
7. Present the implemented compact GNN as the slide-level baseline. The unavailable 120-video/codebase reference is not an executable reproduction requirement; use the complete documented usable inventory. Include mean, Ridge and MLP controls.
8. Quantify every training run: explicit target/loss, internal validation or OOF error, stopping epoch and regularization. Evaluate all main models on exactly the same test videos with log MAE/RMSE and Spearman; add raw-view errors and campaign-level bootstrap uncertainty.
9. Lead with unseen campaign variants. Add leave-one-brand-out evaluation with held-out brands excluded from preprocessing, stopping and fitting. State that this is a separate transfer stress test.
10. Conduct error analysis, compare model definitions fairly, and discuss paid spend, channel audience, exposure age, celebrity/appearance, package selection and noisy extraction as confounders. Do not claim causality, brand recognition or sales effects.

## Demonstration and case studies

Use the trained/frozen GNN to score a new-video storyboard and several held-out existing videos. Each case must contain:

- Brief/objective;
- Creative elements and source evidence;
- Predicted versus observed original response, with clearly labelled uncertainty;
- A transferable recommendation stated as an association or testable hypothesis.

There is no observed outcome for the edited/new video unless a prospective experiment was actually run. Search only training-supported candidates; keep a baseline option. The website uses **Streamlit + Python**, running locally with `python -m streamlit run app.py`. Python directly loads the existing model checkpoint for predictions. It prioritizes evidence-linked recommendations, includes one storyboard, and provides actual graph structure as supporting context.

## Submission format

- **At most four A4 pages**, using the IEEE conference template (`IEEEtran`, `conference`, `a4paper`), including figures and references.
- Present baseline, numerical target, 80/20 protocol, comparison results, edge/node ablations, cross-brand stress test, error analysis and one compact case.
- Keep detailed code, data dictionary, provenance, full metrics, case studies and website outside the four-page PDF.
- Submit the report plus actual session transcript by **12 November**. Local work logs are supplementary, not a fabricated transcript.
- No automatic Blackboard submission or TA email is authorized.

Optional: frozen external official-channel data can be added with full provenance, but this revision does not manufacture an external holdout or a 120-video subset absent from the supplied archive.
