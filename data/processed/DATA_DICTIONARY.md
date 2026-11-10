# Data dictionary and provenance (revision)

## Inventory and original labels

The latest package per brand is authoritative: HSBC v0.2, Hang Seng v0.2, and the sole latest version of each other brand. This yields 219 unique YouTube URLs over 14 brands. MTR actually has 14 rows. Older packages remain unchanged as source archives.

Most original audience metrics are **not** in `video_nodes.csv`. In each brand's `evidence_nodes.csv`, select `kind == engagement_stats`, then JSON-parse `value_json` if present, otherwise `text`. Keys: `views`, `likes`, `comments`, `shares`. Join via `video_id` after stripping optional `video:`. Original coverage: 219 views; 184 non-null likes; no non-null comments. McDonald's additionally has direct views/likes/comments/shares columns in its video table. Null does not mean zero.

`label_provenance.csv` preserves original source evidence IDs, URLs, package and available captured_at fields. These are archived counts, not a common snapshot and not external validation data.

## New YouTube collection

`data/youtube/snapshot.csv` contains one row for each of 219 supplied URLs. Corresponding `<youtube_id>.json` files preserve retrieval details and the exact public metadata snippets used. Collection time is a roughly 75-second window on 9 November 2026 UTC. No authenticated account, cookie export, private API key, video download or access-control bypass is used.

| Field | Definition |
|---|---|
| video_id | Internal reference ID, no `video:` prefix |
| youtube_id / source_url | Canonical video identity / supplied URL |
| retrieved_at_utc | Request start timestamp in UTC, not the original upload time |
| views | Exact integer `originalViewCount` in the current video's primary-info renderer; otherwise exact player videoDetails count where public; never recommendation-card counts |
| likes | Exact count from the current video's like-button accessibility text; abbreviated displays stay missing |
| comments | Exact count if present in initial comments header; none surfaced in this collection. Lazy loading / withheld headers limit coverage. This does not establish zero comments. |
| channel_id / channel_name | Current video's owner renderer; no recommended-channel substitution |
| subscriber_count_display | Public rounded subscriber display, a descriptive string; not an exact label and not used as a predictive feature |
| platform_publish_date | Available exact publication metadata; not exposed in this unauthenticated pass, so original supplied publish_date is retained |
| http_status / status | Request result and extraction outcome, including no_exact_view_count |
| response_sha256 | Digest of the received HTML, to audit retrieval integrity |
| evidence | JSON file only: raw count/display snippets, channel metadata, extractor paths |
| playability_status / reason | Public playback response status; metadata can be present even when playback requires sign-in. No playback barrier is bypassed. |

Fresh coverage: 214 exact views/channel IDs; 175 exact likes; no exact comments. Five URLs without fresh exact views are excluded from the revised supervised cohort. **No historical count is substituted into this fresh cohort.** The original 219 labels remain available separately. No counts are synthesized from fit_score or other annotations.

## Revised modeling table: videos_v2.csv

- `views`: refreshed exact YouTube label; primary target is natural log `ln(1+views)`.
- `archived_views`: untouched prior label, for provenance only; excluded from all model inputs.
- `captured_at`, `retrieved_at_utc`: fresh measurement time.
- `publish_date`, `publish_date_archived`: source-supplied date (platform refresh date unavailable); not a newly verified upload timestamp.
- `age_days`: fresh capture minus supplied upload date, in days; negative/missing values become missing and are imputed inside training only.
- `duration_seconds`: source duration, imputed inside training if missing.
- `brand_family`, `creative_mechanic`, `structure_type`, `evidence_mode`, `cta_explicitness`, `proof_strength`, `pacing`, `production_complexity`: fixed allowlisted categories.
- `main_message`, `hook`, `value_promise`, `cta`: response-sanitized creative text used by the global root encoder.
- `group`: union of normalized same-brand campaign labels and exact same-brand main-message duplicates. This is heuristic grouping, not manually verified campaign lineage.
- Remaining source fields and current engagement/channel columns are audit metadata only unless explicitly allowlisted by the model code.

## Graph provenance

`artifacts/v2/raw_graphs.json`: one response-free graph per labelled video. Allowed creative aspect nodes and capped textual evidence nodes. Memberships/supports are copied from original `candidate_edges.csv`, with schema harmonization; the original `source_edge_id` and package are retained in `edge_provenance.csv`.

The graph is symmetrized for mean message passing. Original support links can be heuristic and are marked. Scores, risk assessments, engagement_stats, source extraction-quality flags, response-bearing sentences and cross-video heuristic similarity suggestions are excluded from the primary per-video graph.

Similarity extension: top two nonroot within-video neighbors with training-fitted SVD-text cosine >=0.55, symmetrized and deduplicated. Threshold, caps and feature rules are specified before evaluation and are not selected from test labels. The node-as-video extension separately uses original same-brand/author/creative category links, with optional five-nearest-training-feature links.

## Split and outputs

`artifacts/v2/splits.csv` contains campaign-group-disjoint outer training (171) and testing (43): 79.91% / 20.09%, the closest rounded 80/20 count for N=214. Three internal grouped folds tune preprocessing/model parameters. Final models refit the entire outer training set for the median chosen fold stopping epoch. All main models share this split and metric definitions.

`metrics.json`: log MAE/RMSE, Spearman, raw-view errors, model selection, selected parameters. `cross_validation.json`: fold RMSE and stopping epochs. `training_histories.json`: final training losses. `leave_brand_out.csv`: separately refitted cross-brand stress tests. OOF residual bands are heuristic sensitivity bands, not guaranteed individual coverage intervals.
