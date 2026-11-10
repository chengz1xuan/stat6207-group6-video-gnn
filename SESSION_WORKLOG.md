# Session work log - 9 November 2026

This is a summary of completed work, not a verbatim transcript. The actual visible-message export is included as output/session_transcript.md.

1. Read the supplied Group 6 project plan. Its instructions were treated as project context, alongside the user's explicit assignment request.
2. At the user's request, switched project commands to the existing Conda `stat6207` interpreter. Project-only VS Code interpreter configuration was written; global defaults were not changed. Optional PDF/Node helpers were installed into this folder, not into Conda.
3. Audited all data schemas. Discovered view counts in engagement_stats evidence for all 219 retained records. Kept latest HSBC/Hang Seng packages; harmonized McDonald's schema; documented MTR's 14 actual rows.
4. Preserved label provenance. Rejected `updated_at` as a substitute for actual view measurement timestamps. This produces 69 missing-age records, handled with training medians plus missingness flags. An earlier preliminary run using an update-time proxy was superseded; no preliminary metrics are final claims.
5. Grouped campaign labels and exact same-brand main-message duplicates; reserved disjoint test/validation groups. Fit preprocessing and all neighbor rules on training data. Tuned hyperparameters using three grouped training folds.
6. Trained the mean predictor, metadata Ridge, full Ridge, MLP and mean-aggregation GraphSAGE. Retained true final results even though the graph model does not clearly outperform the feature-only alternatives.
7. Computed fixed-test bootstrap intervals, three additional GNN initialization runs, test predictions and brand/view-scale error slices.
8. Enumerated constrained creative settings for one illustrative new video and six deterministic test videos. Recorded supporting training examples and explicitly avoided claims of observed or causal uplift.
9. Built the interactive website, exported actual model weights, and checked Python/JavaScript feature and prediction parity for all 219 videos and six optimization cases.
10. Ran headless Chrome checks for prediction, optimization, graph inspection and mobile overflow. Generated and visually reviewed the 10-page PDF report and desktop/mobile screenshots.
11. Prepared reproducible source, dependencies, run manifests, source hashes, model checkpoint and submission packaging. Hosting/deployment status is recorded separately when available.

The actual visible user/assistant transcript was exported from the current local session log, with a documented cutoff and without internal reasoning, tool payloads or credentials. Outstanding user-side action: upload the transcript and final PDF to Blackboard. No submission receipt, email or new-video experiment is fabricated.

## Revision for the additional course/TA requirements

- Replaced the submission report with a genuine four-page A4 IEEEtran conference paper; preserved the old 10-page report only in output/archive as a superseded local version.
- Refreshed all 219 supplied YouTube URLs in one public-metadata collection window. Recovered 214 exact views, 175 exact likes and 214 channel identities; no exact comments surfaced in initial page metadata. Preserved failures and full extraction provenance. Used only the 214 fresh view labels for the new cohort, without archived-count substitution.
- Implemented standard PyTorch Geometric graph-level regression: each video is a creative/evidence subgraph with original candidate_edges memberships/supports. Added fixed within-video semantic similarity as an ablation, and compared the provided structural video-node design with and without feature neighbors.
- Used 171 training / 43 testing videos, a rounded 80/20 grouped split. Three training-only grouped folds tune regularization and stopping; final models refit all outer training rows. All eight main models share the same held-out samples and metrics.
- Recorded explicit regression targets, final training losses, CV metrics and test results. Added campaign-level bootstrap uncertainty and 56 model-brand leave-one-brand-out fits under separate, leak-safe internal stopping.
- Updated six cases to brief → creative elements → predicted/observed original views plus uncertainty → transferable hypothesis. Added a new-video storyboard with declared-input graph only. No observed counterfactual or causal uplift is claimed.
- Exported the actual revised PyG model to the browser. Numerical checks cover all 214 node-feature arrays, identical edge indices, predictions and six optimization results. Browser graph/prediction/optimizer/mobile checks passed.
- Final primary subgraph+similarity log RMSE is 2.238, while the video-node structural extension reaches 2.099. The graph redesign is not described as a proven improvement; small similarity differences and cross-brand weakness are retained in the paper.

## Streamlit + Python implementation

At the user's request, replaced the active web demonstration with a local Streamlit app (`app.py`). It loads the existing frozen PyTorch Geometric checkpoint directly through `src/model_service.py`; model training and reported experiment results are unchanged. The app includes baseline comparisons, LOBO/error tables, actual source/similarity graph inspection, editable prediction and constrained search, supporting training sources, six cases, a new-video storyboard, evidence inspection and report/data downloads.

Streamlit 1.66.1 was already available in Conda stat6207. Added only project-local Streamlit configuration. The server listens at 127.0.0.1:8501 with telemetry disabled. The previous hosted/static prototype is historical local source, not the active app or the new submission package. No hosted site was modified or deleted during this switch.

Verified all 214 Python predictions against the frozen reference (maximum log difference 0), all six optimization cases, existing-graph parity, new-brief graphs without borrowed evidence, and Streamlit AppTest flows for prediction, optimization, edited settings and graph controls (zero app exceptions). Updated the IEEE report's implementation description and verified it remains four A4 pages with no overfull layout boxes. Rebuilt the package around the Streamlit app and refreshed the visible-message transcript.

## 2026-11-10 scientific audit (supersedes the earlier experiment numbers above)

Manually unioned Prudential 170th anniversary Chinese/English variants using a versioned override CSV before grouped splitting. Refit every model on the corrected 171/43 split (seed 50), including fold-local preprocessing/stopping. Added training-only brand log mean with unseen-brand fallback and a node no-brand-edge control. Provenance shows all 1,558 original video-node pairs form complete brand cliques; removing same-brand endpoint pairs leaves zero edges. No claim of creative-relationship efficacy is supported.

Training CV now selects structural-only as the primary subgraph; its corrected log RMSE 2.402 is worse than metadata Ridge 2.375 and video-node brand cliques 2.344. Across the suite, the no-brand-edge control has the lowest tuning CV error. We preserve this distinction and do not select the more favorable similarity model from test scores. Conditional paired intervals do not establish a stable adjacency gain.

Changed the new-brief exposure age from 30 days to the training median (718.5 days); labels remain cumulative snapshot counts. Explicit out-of-support warnings and exported diagnostics mark 30-day inputs as unvalidated extrapolation. Neither in-range ages nor the new sparse brief graph establish a validated forecast.

Preserved the historical 1,019-versus-442,536 third-party McDonald’s reupload failure, with captured uploader, source URL and supplied upload date. The video is now training, and current held-out IKEA failure is reported separately. Added an actual source concept graph and complete corrected test scatter to the four-page IEEE report. Added regression checks for grouping, training-only baseline, edge provenance, age warnings, and source graph fidelity.

The old test set had been inspected before corrections; this is transparently a post-audit corrected internal evaluation, not a new untouched external test. No frozen view labels were altered or videos selectively excluded.
