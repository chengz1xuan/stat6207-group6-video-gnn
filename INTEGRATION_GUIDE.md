# v4 integration guide

The active release is `report/campaign-audit-v4`, based on `e51ce4f` from `mingkun/paired-policy-experiment`. The README is the current detailed guide. Earlier ZIP/patch integration instructions are superseded.

- The report and App default are bound by `artifacts/final_v4.json` to the audited 161/40 split plus 13 quarantined videos.
- `artifacts/v2` remains an immutable historical reference and old-policy input. It is not the new report's evaluation.
- The tracked `artifacts/isolated_revision/final-v4` bundle includes data, source provenance, models, metrics, all 70 LOBO records, complete cases and an experiment dictionary.
- The tracked `artifacts/paired_revision/final-v4` includes both ten-model suites, same-cohort/split controls, four GNN seeds and nonempty output checksums.
- The author-declared brief is separate from extracted observations and bound to a checkpoint hash.
- Package construction includes `Scripts/`; archived package import is checked. Optional bad manifests cannot break the main result selector. Path roots are normalized and temporary cache exclusion uses relative paths.

Run the verification commands in README section 11. Regenerate the PDF only from its measured v4 configuration, not by hand-editing old scores. Older experiment bundles missing required audit outputs are rejected with diagnostics. Do not replace source hashes to force a bundle to load.

No PR merge, Blackboard submission or email is implied by publishing this branch.
