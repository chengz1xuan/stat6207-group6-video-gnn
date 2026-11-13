# Audited v4 verification record

Date: 2026-11-13 (Asia/Hong_Kong). Branch: `report/campaign-audit-v4`, based on `e51ce4f`.
Environment: macOS, existing Conda stat6207 / Python 3.11.16, Torch 2.14.0, PyG 2.8.1, Streamlit 1.66.1.

| Check | Result |
|---|---|
| Unit tests | 57 passed, zero errors/skips; real frozen-cohort checks included |
| Campaign isolation | 13 registered families closed through outer and inner folds; 161 train / 40 test / 13 quarantine |
| Labels | All eligible responses agree with unchanged frozen exact platform labels |
| Full evaluation | Ten-model CV/refit/test and 70 model-brand LOBO runs completed |
| Paired policies | Same cohort/folds/budget; shared-control predictions agree within 1e-6; two GNNs retain four seeds/checkpoints |
| Integrity | Nonempty paired source/output SHA-256 inventory agrees; relative temporary-cache exclusion and normalized Mac roots |
| Inference | All 201 revised and 214 historical predictions pass saved-output checks; six optimizations and declared brief reproducible |
| Streamlit AppTest | New and historical selection, prediction, optimization, state clearing, age warnings and declared inputs pass |
| Real Chrome | Desktop 1440x1000, mobile 390x844, real graph SVG and optimization; zero App/JS exceptions and no page overflow |
| Extracted ZIP | All file checksums agree; Scripts imports, complete model bundle, 201 default, nonempty declared hook and 30-day warning pass |
| PDF | Four A4 pages rendered and visually reviewed; all four supplied names/emails; real graphs, measured tables and complete cases |

The package test initially detected a missing author-brief sidecar; packaging now includes the report-bound path and a regression assertion. Both runtime policy modules and the author-brief file are verified in the extracted package. No local credentials or dependencies are packaged.

Primary subgraph+similarity: CV log RMSE 2.527, training 2.219, testing 2.678; Metadata Ridge testing 2.650. Paired policy intervals include zero. These checks do not establish GNN superiority, a policy benefit, external generalization or causal creative uplift. Campaign coverage is targeted and source/exposure limitations remain explicit.

Machine-readable records are in `artifacts/isolated_revision/final-v4/evaluation/workspace/artifacts/v2/`: `v4_validation.json`, `package_validation.json`, `streamlit_browser_validation.json`. Actual new-model App switching is recorded in `artifacts/isolated_revision/final-v4/evaluation/app_validation.json`.

The final ZIP may be rebuilt after documentation/verification records are included; its SHA-256 file inventory binds the final archived contents. No training code, model or report change is implied by that rebuild.
