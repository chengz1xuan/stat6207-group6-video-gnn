# Isolated campaign, text, graph and evaluation revision

> **整合候选版更新：**下文 `full-app-v3` 数值和“八模型”描述是明坤旧分支的历史记录，
> 不代表本包含有这些产物。本包不含隔离实验结果，实际默认加载最新 main 的正式结果。
> 本版重新执行评估时调用最新 main 的十模型套件；必须在新目录重跑并重新记录数值。
> main 的源码和正式产物未被旧版覆盖。Mac 命令与完整验证范围见根目录
> `INTEGRATION_GUIDE.md`、`INTEGRATION_TEST_REPORT.md`。旧实验如未通过源码哈希校验，
> 不能仅改哈希绕过。本说明后续保留旧工作过程供追溯。

The preparation tools use Python 3.10+ and the standard library only. They are
offline and never modify the original data or `artifacts\v2`.
Training is a separate, explicit operation. The App can now select completed,
isolated exploratory experiments without replacing the original results.

## 本次修改解决的潜在问题

本说明独立维护于 `Scripts\README.md`。根目录 `README.md`、
`src\pipeline.py`、`src\revision.py`、原始数据、原模型和 `artifacts\v2\`
均保持不变。应用接入修改位于 `app.py`、`src\model_service.py`，并新增
`src\experiment_results.py`。

| 潜在问题 | 本次处理 | 仍需注意 |
|---|---|---|
| 同一母活动因中英文名称或版本不同被拆成多个 campaign，可能同时进入训练集和测试集 | 增加显式母活动 ID 和传递合并，例如保诚 170 周年活动；按合并后的整组重新划分，并在分组交叉验证中沿用新 ID | 映射依据为源文本，不代表逐条观看确认；活动清单尚非穷尽人工核验 |
| 部分系列是否属于同一活动尚不确定，强行归组可能引入新错误 | 将 `needs_review` 所在的整个连通活动组隔离，不参与训练、验证或评分 | HSBC One Choice 等待审系列须由人工确认，脚本不会自行判定 |
| 正常创意文本中的 `like`、`share` 被当作播放互动指标，导致整句或节点误删 | 改用带数量的受众反馈识别策略，保留普通创意表达；从原始源节点重建图，而不是仅修改已删减的旧图 | 过滤属于文本规则，不等于完整语义理解 |
| hook 与主信息完全重复，proof 与价值主张重复，摘要又被当作独立证据 | 清空重复 hook 并标记原因，去掉重复 proof；独立证据只保留描述、对白、屏幕文字和片尾文字等观察来源 | 不补造新 hook、proof 或证据；源提取仍可能有噪声 |
| 启发式支持边可能把价格、产品主张连接到不相关对白等内容 | 将未审查的启发式支持边移入审查清单，保留来源和排除理由；仅按既定规则保留直接观察支持边 | 保留下来的边也不宣称全部经过人工语义确认，缺少支持会明确报告 |
| 源文件含 NUL、历史点赞评论与新快照不一致，可能污染读取或混淆字段 | 只在副本中清理 NUL；核对 URL、精确播放量和带时区的采集时间；区分刷新字段和历史归档字段 | 不重新采集，不宣称重新核实了平台发布日期；乱码名称及可疑年份仍保留待审 |
| 修正后继续沿用旧划分、旧模型或旧过滤函数，会使新结果名不副实 | 新建整组划分；隔离重新拟合和评估；App 显式选择结果版本，新模型加载和编辑预测沿用修正后的文本策略 | 完整训练必须显式启动；新旧样本、图和划分不同，分数不能作为配对提升比较 |
| 修正流程可能意外覆盖原始文件，或使用已被修改的中间产物 | 输出限定在新的隔离目录，拒绝覆盖已有运行目录，并校验输入、输出和准备策略的哈希 | 生成的数据与评估结果不会随代码自动上传，也不会覆盖默认应用结果或正式报告 |
| App 显示旧样本数、旧下载文件或上一次实验的预测，却加载了新模型 | 按所选实验动态展示样本、划分、案例和指标；切换时清除预测与编辑状态，原版 PDF 只在原版模式提供 | 默认仍为原版；新实验必须完整导出并通过哈希校验后才出现在选择器 |

**交付范围：**包含离线修正、完整重新评估和 App 版本切换。某次运行是否已完成，
以该运行目录中的 `evaluation\summary.json` 和最后发布的
`evaluation\workspace\artifacts\v2\app_manifest.json` 为准；仅有训练中间文件不代表完成。
本流程不宣称全部数据问题已人工解决，也不据新旧分数宣称模型改进。
所有导出仍标记为 `training_ready: false`，完整评估需要明确接受待审限制。

| 文件 | 职责 |
|---|---|
| `review_rules.json` | 母活动映射、待审状态、指定排除边和人工检查事项 |
| `isolate_revision.py` | 离线核对、全局分组、样本或全量导出及新划分 |
| `text_policy.py` | 数据准备和评估共用的修正文本过滤策略 |
| `graph_revision.py` | 从原始来源重建图并记录证据、节点和边的处理 |
| `evaluate_revision.py` | 默认只读计划、轻量均值检查及显式完整重新评估 |
| `test_isolate_revision.py`、`test_revision_workflow.py` | 隔离、分组、文本、图、清单校验和评估入口的回归检查 |
| `run_experiment.py` | 一键执行全量准备、完整重新评估及 App 结果发布；默认仅显示计划 |
| `requirements-experiment.txt` | 独立实验环境的依赖范围，不替代原项目版本清单 |
| `test_app_bundles.py` | 完成标记、结果发现、哈希防篡改与显式执行保护的回归检查 |
| `validate_app_experiment.py` | 核对全队列推理一致性、旧模型预测及真实 Streamlit 切换 |
| `run_paired_experiment.py` | 固定修正后的样本与划分，隔离重训 v2/v3 两套处理策略，输出配对统计 |
| `test_paired_experiment.py` | 双策略字段恢复、控制变量、来源哈希、失败保护与配对统计回归检查 |

## 双策略公平重训（新增）

此入口不是将两份历史分数相减，也不是重复运行两次 v3。它读取一个完整的
`isolate_revision.py --full` 准备目录，以及正式
`data\processed\videos_v2.csv`、`artifacts\v2\raw_graphs.json`：

- 两组共同使用准备目录中所有非隔离视频、冻结标签、修正后的活动组及原定
  train/test 划分；共同固定训练内三折、品牌/时长/年龄等输入和视频节点关系。
- A (`v2`) 恢复原表的四个文本字段，使用旧文本过滤器及旧子图；
  B (`v3`) 使用修正字段、数量型受众反馈过滤器及保守重建子图。
  会核对 `original_*` 来源，不能把已经删空 hook 的 v3 数据当成 A。
- 分别重训现有十模型套件；预处理只在各训练折拟合，两组调参预算相同。
  结构子图 GNN 和相似边 GNN 均使用 42、7、21、84 四个初始化种子。
  种子 42 完成训练内调参，其余种子固定所选正则化及轮数，仅检查初始化敏感性。
- 两组从独立 Python 子进程加载隔离源码，防止新版文本策略污染旧版。
  原表、源码、检查点及旧指标均不修改。任一步失败不发布完成的 `summary.json`。

在项目根目录执行。以下示例使用本地已存在的 `full-app-v3`；若目录缺失，
先用前述 `isolate_revision.py --full --output <新目录>` 准备，不需要先单独训练 v3。

默认只核对输入并显示计划，不创建输出目录、不训练：

```powershell
.\.venv\Scripts\python.exe Scripts\run_paired_experiment.py --input artifacts\isolated_revision\full-app-v3 --output artifacts\paired_revision\v2-v3-paired-01
```

可先使用一个独立输出名检查数据链路。此模式只计算共同训练均值，**不是模型比较**：

```powershell
.\.venv\Scripts\python.exe Scripts\run_paired_experiment.py --input artifacts\isolated_revision\full-app-v3 --output artifacts\paired_revision\v2-v3-smoke-01 --smoke
```

显式执行完整双策略重训：

```powershell
.\.venv\Scripts\python.exe Scripts\run_paired_experiment.py --input artifacts\isolated_revision\full-app-v3 --output artifacts\paired_revision\v2-v3-paired-01 --run --acknowledge-exploratory
```

输出必须是 `artifacts\paired_revision\` 下的**全新子目录**，不能复用 smoke 或失败运行目录。
依赖使用项目现有科学计算环境；Mac/Linux 将解释器替换为 `.venv/bin/python`，
并使用当地路径写法。此入口不重新采集、不重复 LOBO，不发布 App bundle，也不覆盖正式 PDF。

| 输出 | 内容 |
|---|---|
| `protocol.json` | 两套策略、固定条件、种子、输入/源码 SHA-256 和解释边界 |
| `common\splits.csv`、`common\inner_folds.csv` | 两组共同的样本、标签、外层及内层划分；smoke 不生成内层折 |
| `v2\workspace\artifacts\v2\`、`v3\workspace\artifacts\v2\` | 各自重新测得的指标、CV、逐条测试预测、GNN 检查点及种子预测 |
| `comparison.csv`、`comparison.json` | 种子 42 十模型配对结果，以及两种子图 GNN 的逐种子配对结果 |
| `summary.json` | 仅成功结束后生成的完成记录与结果哈希；检查 `mode`，smoke 不代表完成训练 |

先看同名模型的 `delta_log_RMSE`、`delta_log_MAE` 及各自 `_95` 区间：
差值定义为 **v3 减 v2**，负误差差值有利于 v3。区间使用相同测试活动组的
2,000 次配对 bootstrap；区间跨零则不能称为明确改善。CSV 将区间拆为
`_95_low`、`_95_high` 列。另提供 Spearman；原始播放量误差保留在各组 `metrics.json`。
均值、品牌均值、Metadata Ridge 的逐条预测必须一致（绝对容差 1e-6），
否则比较终止，不能生成成功结果。

这是已经检查过测试集后的**内部配对复核**，区间仅以当前划分和拟合为条件，
不覆盖选模、划分或所有随机性。它估计文本与图清理的整体策略差异，
不能单独归因于删 hook、去边等某一项，也不能宣称创意修改的因果提升。
不得根据此次测试分数反复修改规则或挑选最有利种子。

## 0. 完整实验与 App 接入

在仓库根目录准备独立环境。下面使用已安装的 `uv` 和 Python 3.12，
不修改全局 Python 环境：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend cpu -r Scripts\requirements-experiment.txt
```

使用一个**尚不存在**的输出目录完成整个实验：

```powershell
.\.venv\Scripts\python.exe Scripts\run_experiment.py --output artifacts\isolated_revision\full-app-v3 --run --acknowledge-exploratory
```

不带 `--run` 时仅显示执行计划，不导出数据或训练。显式执行依次完成：
全量离线核对与图重建、母活动整组划分、八模型三折训练内调参与重训、
测试指标及 2,000 次活动组 bootstrap、折外残差及种子敏感性、
四模型逐品牌留出压力测试，以及新模型、预测案例、App 数据和来源记录导出。
中间任何步骤失败都返回非零退出码，不将未完成结果发布给 App。

完成后核对实际推理与界面切换，并启动应用：

```powershell
.\.venv\Scripts\python.exe Scripts\validate_app_experiment.py --input artifacts\isolated_revision\full-app-v3 --check-app
.\.venv\Scripts\python.exe -m streamlit run app.py
```

在左侧 **结果版本** 中选择“探索性修正实验 · full-app-v3”。应用启动默认
保留原版，不合并两套结果。新模式展示实际纳入的训练/测试规模、待审隔离提示、
新模型指标、重建图和重新生成的案例；下载指标也对应当前选择。
切回原版不会保留新模式的预测或表单编辑。

`src\model_service.py` 对新检查点使用独立的特征恢复策略，确保 `like`、
`share` 修正同时用于训练和在线预测，而不修改旧模型的特征方法。
新结果必须具有完整的 `app_manifest.json`，且检查点、指标、案例、输入清单、
评估来源和文本策略哈希一致；校验失败会明确报错，不偷偷回退到旧结果。
完整实验的实际依赖版本记录在 `evaluation\provenance.json`；
推理与界面核验记录在 `evaluation\app_validation.json`。

### 本次实际实验：full-app-v3

2026-11-10 在当前工作区完成全量离线准备、八模型评估及 14 个品牌的
逐品牌留出压力测试。214 条视频中，161 条训练、40 条测试、13 条待审隔离。
App 的 201 条纳入视频预测与训练端逐条一致；原版预测保持不变。

| 当前测试集模型 | Log MAE | Log RMSE |
|---|---:|---:|
| Mean | 2.252 | 2.713 |
| Metadata Ridge | 1.891 | 2.483 |
| Subgraph structural | 2.165 | 2.656 |
| Subgraph + similarity（训练内选择的主子图模型） | 2.180 | 2.644 |

该测试划分下，主子图 GNN **没有优于 Metadata Ridge**。这些值属于新的
探索性样本/图策略/划分，不能与旧实验分数直接解释为性能提升或下降。
完整八模型指标、区间、折外预测和跨品牌结果均在本地运行目录中。

`full-app-v3` 为本机生成的运行目录，受 Git 忽略规则保护，不随代码自动上传。
新克隆的仓库只会显示原版，须先运行上述完整流程生成新结果。
当前工作区已存在该目录，重复实验时必须改用新的 `--output` 名称，
并在验证命令中使用相同的新名称。

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
they are **not a paired improvement comparison**. These outputs become a separate selectable App experiment only after all exports
complete. They never replace the original default, final PDF, old cases or scores.

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
| `evaluation\summary.json` | Completed training/evaluation summary |
| `evaluation\provenance.json` | Prepared manifest hash, training source hashes and package versions |
| `evaluation\workspace\artifacts\v2\app_manifest.json` | Completion marker and hashes for App discovery |
| `evaluation\app_validation.json` | Explicitly executed inference parity and optional App checks |

Generated runs are ignored by Git. Never paste sample partitions into the full
cohort, or replace `data\processed`/`artifacts\v2` with isolated outputs.

## Targeted tests

```powershell
python -m unittest discover -s Scripts -p "test_*.py"
```

Tests cover campaign unions/quarantine, ordinary like/share text, real source
regressions, graph reconstruction, evaluation source isolation, train-only smoke
prediction, manifest tampering, completed-bundle discovery and overwrite guards.
They do not train a GNN. `validate_app_experiment.py` additionally requires a
successfully completed full experiment and the scientific/UI environment.
