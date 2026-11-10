# Isolated campaign, text, graph and evaluation revision

The preparation tools use Python 3.10+ and the standard library only. They are
offline and never modify `src`, the original data, `artifacts\v2`, or the app.
Training is a separate, explicit operation with the existing scientific packages.

## 本次修改解决的潜在问题

本说明独立维护于 `Scripts\README.md`。根目录 `README.md`、原有 `src\`、
数据、模型和 `artifacts\v2\` 均保持不变。

| 潜在问题 | 本次处理 | 仍需注意 |
|---|---|---|
| 同一母活动因中英文名称或版本不同被拆成多个 campaign，可能同时进入训练集和测试集 | 增加显式母活动 ID 和传递合并，例如保诚 170 周年活动；按合并后的整组重新划分，并在分组交叉验证中沿用新 ID | 映射依据为源文本，不代表逐条观看确认；活动清单尚非穷尽人工核验 |
| 部分系列是否属于同一活动尚不确定，强行归组可能引入新错误 | 将 `needs_review` 所在的整个连通活动组隔离，不参与训练、验证或评分 | HSBC One Choice 等待审系列须由人工确认，脚本不会自行判定 |
| 正常创意文本中的 `like`、`share` 被当作播放互动指标，导致整句或节点误删 | 改用带数量的受众反馈识别策略，保留普通创意表达；从原始源节点重建图，而不是仅修改已删减的旧图 | 过滤属于文本规则，不等于完整语义理解 |
| hook 与主信息完全重复，proof 与价值主张重复，摘要又被当作独立证据 | 清空重复 hook 并标记原因，去掉重复 proof；独立证据只保留描述、对白、屏幕文字和片尾文字等观察来源 | 不补造新 hook、proof 或证据；源提取仍可能有噪声 |
| 启发式支持边可能把价格、产品主张连接到不相关对白等内容 | 将未审查的启发式支持边移入审查清单，保留来源和排除理由；仅按既定规则保留直接观察支持边 | 保留下来的边也不宣称全部经过人工语义确认，缺少支持会明确报告 |
| 源文件含 NUL、历史点赞评论与新快照不一致，可能污染读取或混淆字段 | 只在副本中清理 NUL；核对 URL、精确播放量和带时区的采集时间；区分刷新字段和历史归档字段 | 不重新采集，不宣称重新核实了平台发布日期；乱码名称及可疑年份仍保留待审 |
| 修正后继续沿用旧划分、旧模型或旧过滤函数，会使新结果名不副实 | 新建整组划分；在隔离工作区使用修正后的文本策略重新拟合和评估，不复用旧检查点 | 完整训练必须显式启动；新旧样本、图和划分不同，分数不能作为配对提升比较 |
| 修正流程可能意外覆盖原始文件，或使用已被修改的中间产物 | 输出限定在新的隔离目录，拒绝覆盖已有运行目录，并校验输入、输出和准备策略的哈希 | 生成的数据与评估结果不随本次代码推送上传，也不会自动更新应用或正式报告 |

**交付范围：**本次交付的是修正与重新评估脚本及独立说明。此前仅生成过
12 条视频的样本，并运行训练均值的轻量流程检查；尚未执行全量修正导出、
全量 GNN 训练或全量评估。因此本说明不宣称全部数据问题已人工解决，
也不提供修正后模型优于旧模型的结论。所有导出仍标记为
`training_ready: false`，完整评估需要明确接受待审限制。

| 文件 | 职责 |
|---|---|
| `review_rules.json` | 母活动映射、待审状态、指定排除边和人工检查事项 |
| `isolate_revision.py` | 离线核对、全局分组、样本或全量导出及新划分 |
| `text_policy.py` | 数据准备和评估共用的修正文本过滤策略 |
| `graph_revision.py` | 从原始来源重建图并记录证据、节点和边的处理 |
| `evaluate_revision.py` | 默认只读计划、轻量均值检查及显式完整重新评估 |
| `test_isolate_revision.py`、`test_revision_workflow.py` | 隔离、分组、文本、图、清单校验和评估入口的回归检查 |

## 1. Prepare corrected data (sample by default)

From the repository root in PowerShell:

```powershell
python Scripts\isolate_revision.py
```

Default: **12 fixed videos across four brands**, output to the new directory
`artifacts\isolated_revision\sample-v2\`. Shared CSV/JSON containers are read, but
only selected videos are rebuilt/exported. Campaign components are calculated
over the complete metadata inventory so IDs remain consistent across samples.
Existing run directories are never overwritten.

To explicitly prepare the full frozen cohort, without training or recollection:

```powershell
python Scripts\isolate_revision.py --full --output artifacts\isolated_revision\full-review-v2
```

Use another fresh `--output` name for repeated runs. Old `sample` exports use a
different schema and cannot be fed to the new evaluator.

## 2. Campaign IDs and unresolved review

`review_rules.json` records stable parent IDs, source video identities, reasons
and review status. `text_evidence` means a conservative series boundary supported
by the source text, **not** verified identical footage or an external human sign-off.
`needs_review` quarantines the entire connected campaign component.

Text-supported groups include IKEA Complete Sleep, For Love storage and Proposal;
McDonald's Shogun Tamago and World Cup WOW sauce; Prudential 170th anniversary;
and Mannings Everyday Wellness. Existing groups are preserved and aliases merged
transitively, including train-only aliases relevant to internal validation.

HSBC One Choice, Kee Wah's 2022 New Year craft series, Prudential's 2023 family
series, Hang Seng Invest Express and Mannings Say Yes to Better remain pending.
Read the source evidence before changing a rule's status. Do not equate repeated
products, general brand slogans, or advertisements from different years.

The registry is not an exhaustive manual campaign inventory. Other components
retain their original grouping plus exact descriptive matches. Each export is
therefore marked `training_ready: false`, even for the full cohort.

## 3. Text and evidence corrections

- Check URL identities, exact frozen views and timezone-aware capture times
  against the source row, snapshot and collection JSON. Disagreement fails.
- `text_policy.py` removes sentences with quantified audience responses, not
  isolated words. Ordinary `ingredients like fresh eggs`, `share happiness`,
  decimal numbers, URLs and author-declared like-page CTAs survive.
- Preserve original creative text in `original_*` fields. A hook copied verbatim
  from the main summary is blanked and marked `duplicate_summary`; do not invent
  an opening hook. `likes`/`comments` are refreshed fields; historical values are
  explicitly preserved under `*_archived`.
- Rebuild graphs from **original source nodes and candidate edges**, rather than
  editing already-filtered frozen graphs. This restores incorrectly removed
  ordinary creative text.
- Independent evidence kinds are source descriptions, extracted spoken dialogue,
  on-screen text and end-card text. Summaries/CTAs/audience takeaways are not
  treated as independent evidence nodes. Extraction can still be noisy.
- Omit a proof annotation that merely repeats the value promise. Unreviewed
  heuristic support links are quarantined in the edge ledger; direct deterministic
  observation links may remain. No automatic semantic claim or support is invented.
- Preserve original edge IDs/endpoints, record duplicate-node remapping, and merge
  parallel edges while retaining additional provenance. Fixed pre-evaluation
  caps: three observations per kind, one aspect per field, four visual/audience
  signals. Unsupported/missing support is reported explicitly.
- Clean NULs only in copies, including encoded `value_json`. Source-file NUL counts
  describe the scanned file, not just exported rows.
- Uncertain creative labels, suspicious campaign years and corrupted-looking
  product names remain unchanged in a manual review queue.

This is deliberately a different, conservative graph policy, not a claim that all
evidence has been watched or human-verified.

## 4. New split and evaluation

Eligible videos are repartitioned by whole campaign. Seed 42 and a subset-sum
calculation select the closest possible rounded 20% test count using group sizes,
never response labels. Quarantine is excluded from fitting, reference graphs,
cross-validation and scoring. Old split/group fields remain available for audit.

Inspect the evaluation plan without importing scientific libraries or training:

```powershell
python Scripts\evaluate_revision.py --input artifacts\isolated_revision\sample-v2
```

Run a small, standard-library-only data-path check:

```powershell
python Scripts\evaluate_revision.py --input artifacts\isolated_revision\sample-v2 --smoke
```

This produces `evaluation-smoke\summary.json`, using only the training mean.
It is **not a GNN evaluation or a model-quality result**.

Full refitting requires a full-cohort prepared export and explicit acknowledgement
of unresolved manual reviews and the exploratory nature of the results:

```powershell
python Scripts\evaluate_revision.py --input artifacts\isolated_revision\full-review-v2 --run --acknowledge-exploratory
```

Use an environment with the project's NumPy, Pandas, SciPy, scikit-learn,
Matplotlib, PyTorch and PyTorch Geometric dependencies already installed. No
automatic dependency installation is performed.

The evaluator stages copies of `src\pipeline.py` and `src\revision.py` under
`evaluation\workspace\src`, replacing only their text-cleaning functions with the
shared corrected policy. The original source modules never execute. It then
passes the prepared cohort, reconstructed graphs, source video relations and
**already frozen new split** into the original `train_all` implementation:

- All eight model definitions, three training-only grouped folds, original
  hyperparameter search, stopping/refitting, log/raw-view metrics;
- Training-only preprocessing and model selection, campaign bootstrap intervals,
  OOF predictions/residuals, seed sensitivity and a newly trained checkpoint;
- Separate leave-one-brand-out evaluation by default. Explicitly pass
  `--skip-brand-transfer` to omit that expensive stress test.

Results are in `evaluation\workspace\artifacts\v2\`, with a top-level
`evaluation\summary.json` and source/input provenance. The evaluator does not call
the old data-loading, graph-building or split-selection entry points.

Evaluation refuses modified exports or changed preparation policies. No existing
checkpoint is reused. Old and new metrics have different samples/graphs/splits:
they are **not a paired improvement comparison**. These outputs do not update
the app, final PDF, old cases or reported scores.

## Export files

| File | Meaning |
|---|---|
| `audit.json` | Scope, policy, limitations, review queue and applied campaigns |
| `manifest.json` | Input/output hashes and preparation script versions |
| `review_rules.json` | Exact campaign/review registry used for this run |
| `videos.csv` | Corrected selected rows, parent IDs, new and original partitions |
| `proposed_splits.csv` | New whole-campaign split, including `quarantine` |
| `quarantined_videos.csv` | Campaign components awaiting confirmation |
| `raw_graphs.json` | Reconstructed response-free graphs |
| `video_pairs.json` | Source video relations restricted to eligible references |
| `edge_changes.csv` | Excluded/review-pending support edges and reasons |
| `source_sample\` | Cleaned archival source rows, not the revised model graph |

Generated runs are ignored by Git. Never paste sample partitions into the full
cohort, or replace `data\processed`/`artifacts\v2` with isolated outputs.

## Targeted tests

```powershell
python -m unittest discover -s Scripts -p "test_*.py"
```

Tests cover campaign unions/quarantine, ordinary like/share text, real source
regressions, graph reconstruction, evaluation source isolation, train-only smoke
prediction, manifest tampering and overwrite guards. They do not train a GNN.
