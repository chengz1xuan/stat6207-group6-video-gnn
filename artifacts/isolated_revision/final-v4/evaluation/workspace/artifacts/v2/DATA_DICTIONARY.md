# Audited cumulative-view experiment data dictionary

This experiment uses 161 training / 40 testing videos and 13 quarantined videos from the 214-label snapshot.

`views`: exact frozen cumulative YouTube views at 2026-11-09 capture; target ln(1+views).

`group` / `campaign_id`: original groups transitively unioned with source-text-supported parent-series rules. `needs_review` families are quarantined as complete components. `original_group` / `original_split`: provenance only. `split`: train, test or quarantine.

`main_message`, `hook`, `value_promise`, `cta`: quantified-response-free text; duplicate summary hooks are blanked. `original_*`: unedited text for the paired old-policy control.

`likes` / `comments`: fresh platform values where available; missing is not zero. `likes_archived` / `comments_archived`: historical values, not model inputs.

`age_days`: capture minus source-supplied publication date, not a newly verified upload date. Training support 36.5016--5620.5016 days; default new-brief age 672.5013 days. No fixed-horizon or 30-day forecast has been validated.

CAT fields are the eight allowlisted brand/creative categories; duration is in seconds. Channel, engagement and original-text audit columns are excluded from predictive features except the explicitly allowlisted text/category/duration/age inputs.

Structural edges are original memberships and deterministic observation-support links. Similarity links are training-fitted, heuristic within-video top-2 cosine >=0.55. Video-node core has 1382 pairs, 1382 same-brand; 0 remain after removing brand pairs.

Evaluation is post-audit internal evidence, not an untouched external holdout. Campaign coverage and source annotations are not exhaustively video-verified. Creative recommendations and OOF sensitivity bands do not demonstrate causal uplift.
