# Advertising Video GNN — audited v4

本分支 `report/campaign-audit-v4` 基于 `mingkun/paired-policy-experiment`，把活动分组审计、独立重训、双策略配对实验、网页和新版报告绑定到同一个冻结版本。

| 作者 | 邮箱 |
|---|---|
| CHENG, Zixuan | 1155280337@link.cuhk.edu.hk |
| HUANG, Ze | 1155280222@link.cuhk.edu.hk |
| SONG, Mingkun | 1155276943@link.cuhk.edu.hk |
| HOU, Kaiwen | 1155280688@link.cuhk.edu.hk |

[新版四页 IEEE A4 报告](output/pdf/Group6_Final_Report.pdf) · [实验版本配置](artifacts/final_v4.json) · [审计规则](Scripts/review_rules.json)

## 1. 当前版本解决了什么

- 将已识别的 IKEA Complete Sleep、麦当劳 Shogun Tamago / WOW sauce、保诚周年及其他系列按母活动合并，增加美心 2023 贺年广告两篇的统一 ID。
- 保留原分组的传递关系，只合并不拆散；五类尚不明确的活动整组隔离。13 个规则家庭包含八个文本支持系列和五个待审系列，不宣称逐条观看验证或穷尽全部活动。
- 从 214 个精确播放量标签得到 **201 条纳入、13 条隔离**；固定分组划分 **161 训练 / 40 测试**，约 80/20。外层和训练内三折均检查活动隔离。
- 重训十模型，完成 70 个模型—品牌 LOBO 评估，并在完全相同队列/划分上重训旧、新两套文本与图策略。
- 报告写入真实结果、训练 CV、具体优化设置、来源、数值化残差范围、真实概念支持/相似边图及完整测试散点图；作者改为上述四位姓名和邮箱。
- 修复 Mac 路径归一化、临时目录下哈希清单为空、损坏可选 manifest 导致主页面失败、交付包遗漏 `Scripts`、新实验遗漏品牌边解释和错误下载旧数据字典等问题。

**证据边界：**这是检查过旧测试结果后的内部审计修正版，不是新外部 holdout。已识别的活动系列已修正，但活动谱系与创意标签仍非穷尽人工确认。GNN 或清理策略都没有被证明优于简单对照；推荐只是假设。

## 2. 最快运行网页

使用本机已有 Conda 环境，不改变全局默认 Python：

```bash
conda activate stat6207
python -m streamlit run app.py
```

访问 `http://localhost:8501`。应用由本地 **Streamlit + Python / PyTorch Geometric** 实现，直接读取冻结模型，不使用 ChatGPT Sites 或旧托管网页。

本项目在 Python 3.11.16、Torch 2.14.0、PyG 2.8.1、Streamlit 1.66.1 的本地环境核验。首次在其他机器运行时，在自己的项目环境安装：

```bash
python -m pip install -r requirements.txt
```

`.project-deps/` 是当前机器的可选依赖目录，不上传、不打包；其他机器按依赖清单安装即可。无需重新抓取数据或训练才能查看冻结结果。

## 3. 网页内容与实验版本

左侧“结果版本”默认选择 v4 完整实验。`artifacts/v2/` 的历史结果仍可选择，用于追溯，不能把不同队列的分数直接相减。

| 页面 | 内容 |
|---|---|
| 研究结果 | 当前十模型、品牌均值和去品牌边对照、条件 bootstrap、LOBO、误差切片及当前最大测试误差 |
| 图结构浏览 | 实际源节点/边、原始来源 ID、可选显示启发式相似边、节点文本与原视频链接 |
| 预测与优化建议 | 新声明 Brief 或已有视频编辑；直接运行冻结 GNN；受训练支持约束的分类设置搜索；JSON 导出 |
| 案例与 Storyboard | 具有非空作者文案的新 MTR 假设案例，以及六个固定测试视频的完整优化案例 |
| 数据与报告 | 219 链接快照、逐条采集证据、与当前实验匹配的数据字典和指标下载 |

切换实验会清除旧预测及表单状态。报告下载只对其绑定的 v4 实验显示；历史模式不会误下载新版报告。图上的相似边显示开关只影响展示，不替换训练模型。

新 Brief 的默认文案是作者提出的假设，不是从另一个视频借来的证据；图里没有伪造的 extracted-evidence 节点。实际画面、产品信息和当前时刻服务资料仍需制作前核验。

## 4. 标签和时间解释

- 原始最新品牌包共有 219 个 URL，冻结采集时间为 **2026-11-09 12:01:03–12:02:18 UTC**。
- 214 条精确累计播放量和频道身份，175 条精确点赞数；评论计数未取得，保留缺失，不补零。
- 缺少新精确播放量的五个 URL 不用历史计数代替。标签、时间戳、响应摘要与抽取证据保留在 `data/youtube/`。
- 原始 219 views / 184 likes 主要位于各包 `evidence_nodes.csv` 的 `kind=engagement_stats`，解析 `value_json` 或 `text`；它们是历史来源，不替换本轮标签。
- `age_days` 是采集时间减去**源表日期**，不是本次重新验证的上传日期。

v4 训练年龄范围 **36.5–5620.5 天**，中位数 **672.5 天**。新 Brief 默认使用训练中位数；**30 天属于范围外推**，页面和 JSON 均明确标记。即使年龄在范围内，也没有验证新视频在该时间点的播放量。目标是累计快照计数，不是固定 30 天窗口、未来增长曲线或流量承诺。

## 5. 活动分组与隔离

规则文件：`Scripts/review_rules.json`。来源视频的原始分组保持完整，再按源文本支持的母系列做传递合并；同组不能同时参与训练与测试，也不能跨训练内部拟合/验证角色。

特别修复：

- IKEA“睡好覺做好夢”三个版本；麦当劳将军汉堡系列与 WOW sauce；保诚 170 周年母系列。
- 美心 `CxbnjaET7ew` / `WBMGLbAvnQY`：同一“祝福 就是最足料的幸福”2023 贺年系列、同日发布，产品名称不同不会再拆组。
- HSBC One Choice、奇华 2022 贺年等五个 `needs_review` 家庭的完整连通分量共 13 条隔离，不参与拟合、调参、评分或支持来源。

全量准备表保存 `original_group` / `original_split`、新 `campaign_id` / `group` / `split`。固定 seed 42 的整组 subset-sum 只按组大小选择最接近 20% 的测试条数，不读取播放量。最终 201 条样本属于 175 个组；161/40 = 80.10%/19.90%。

待审创意字段不猜测替换，保留审查记录；这与已隔离的不确定母活动是两类限制。

## 6. 图与模型

主设计：每个视频为一个子图，含 context root、创意概念及观察文本证据。v4 排除派生摘要证据、未审查启发式支持边、响应统计及风险/评估分数。保留源 membership 与 deterministic observation-support，原边 ID 可追溯；不补造关系。

root 输入为品牌/七个创意类别、log 时长/年龄及缺失标志、16 维 TF-IDF/SVD 文本。其他节点含类型/字段及独立的 16 维文本表示。词表、编码、填补、缩放和 SVD 均只在相应训练折拟合。

两层 24-unit mean GraphSAGE 后，拼接 root 和全图 mean embedding，再预测标准化 ln(1+views)。相似边扩展使用非 root 节点 top-2、训练拟合 SVD cosine ≥0.55，并标为启发式。

十种比较模型包括常数 log 均值、训练品牌 log 均值、Metadata Ridge、Pooled Ridge、无消息 MLP、两种子图 GNN、两种视频节点 GNN和去品牌边视频节点对照。

**品牌审计：**视频节点结构图的 1,382 对关系全部是同品牌，恰好组成品牌内完整图。去同品牌端点连接后剩零边，但保留品牌输入、相同架构和调参预算。因此此对照隔离邻接作用，不能声称移除了全部品牌信息；视频节点成绩也不能证明概念关系有效。

## 7. 训练与定量评价

共同目标为自然对数 `ln(1+views)`。神经模型最小化训练标准化目标的 MSE，Adam lr 0.01。训练内 3 折 GroupKFold 搜索 Ridge alpha `{10,100}` / 神经 weight decay `{0.01,0.1}`；最多 260 轮、早停 patience 35。平均折 RMSE 选择配置，所选折最佳 epoch 中位数用于完整训练集重训，测试不选择配置。

主子图由训练 CV 选择 **Subgraph + similarity**：CV 2.527，对比 structural 2.535；最终 21 epochs、weight decay 0.1、训练 Log RMSE 2.219。整体套件 CV 最好为视频节点品牌图（2.455），不因此改写预先指定的主设计，也不按测试分数换成 structural。

| v4 模型 | Test Log MAE | Test Log RMSE | Spearman |
|---|---:|---:|---:|
| Mean | 2.385 | 2.805 | 不定义 |
| Brand mean | 2.301 | 2.900 | 0.133 |
| Metadata Ridge | 2.162 | 2.650 | 0.352 |
| Pooled Ridge | 2.200 | 2.743 | 0.346 |
| Pooled MLP | 2.332 | 2.703 | 0.379 |
| Subgraph structural | 2.201 | 2.644 | 0.419 |
| Subgraph + similarity | 2.161 | 2.678 | 0.410 |
| Node structural | 2.217 | 2.723 | 0.357 |
| Node + similarity | 2.261 | 2.812 | 0.266 |
| Node no-brand edges | 2.312 | 2.804 | 0.162 |

主子图测试 RMSE 2.678，高于 Metadata Ridge 2.650，不能宣称优于基线。核心子图的较低测试点误差也不能用于反选主模型。2,000 次 campaign bootstrap 为固定划分/模型条件区间，不包含划分、规则选择或所有随机性。

LOBO 在 201 条纳入数据中逐品牌留出、其余 13 品牌内选择停止轮次，再重训；固定 alpha=100 / weight decay=0.1，五模型×14品牌=70个评估。主子图宏平均 Log RMSE 2.696，Pooled Ridge 2.579，未证明跨品牌优势。

## 8. 同队列双策略配对比较

两套策略共同使用同一批201条视频、冻结标签、已修正组、外层划分、训练内折、视频节点关系、模型和调参预算。旧策略恢复未清理的四个文本字段及冻结旧子图；新策略用清理字段与保守重建图。均值、品牌均值、Metadata Ridge 的逐条预测必须在 1e-6 内一致。

两种子图均保留 seed 42、7、21、84 的模型与预测；只有42做调参，其他种子固定已选正则化/轮数。下表是 seed 42：

| 模型 | 旧策略 RMSE | 新策略 RMSE | 新减旧的95%配对区间 |
|---|---:|---:|---|
| Subgraph structural | 2.703 | 2.644 | [-0.158, 0.039] |
| Subgraph + similarity | 2.696 | 2.678 | [-0.186, 0.126] |

区间都跨零，未建立清理策略改善。该比较估计文本/图处理整体策略和学习流程的差异，不能归因于单独删 hook 或某条边，更不是创意因果效果。历史171/43结果不直接与v4相减。

## 9. 具体案例与不确定性

六个已有视频按固定测试索引前六条选择。搜索最多改变 mechanic/structure/pacing 中两项，候选三元组须在训练出现至少两次，支持例子只能来自训练；没有候选提高估计则保留原设置。

- 恒生案例 `-gNF3ekne-U`：observed 1,659,191，原预测 1,016,459，建议预测 1,267,737；Testimonial → Problem-solution，medium tempo → fast-cut。原预测敏感度范围约19,029–54,292,592，不能保证个体覆盖，更不是uplift区间。
- 作者声明的30秒MTR假设Brief：保持旅行规划语境，Demo → Problem-solution，medium tempo → fast-cut；原预测316,494，建议580,104。原敏感度范围约5,924–16,905,092；没有实测新视频结果。文案不宣称当前班次、耗时或实际服务情况。

范围来自训练OOF绝对log残差的名义90%分位，不是校准好的个体预测区间。跨产品/品牌的训练例子仅支持分类组合，不能保证迁移。类别变化与固定源证据可能不一致，必须重写成连贯脚本，再以可比媒体曝光验证。人物吸引力、频道/受众、广告投放、时间、转载身份与抽取噪声都可能解释结果；不声称销售、品牌认知或因果提升。

## 10. 文件与版本绑定

| 路径 | 用途 |
|---|---|
| `AUTHORS.json` | 四位报告作者与邮箱 |
| `artifacts/final_v4.json` | App默认、报告、主实验、配对实验与作者Brief绑定 |
| `artifacts/isolated_revision/final-v4/` | 本次全量准备数据、规则、分组、隔离表、source_sample与输入哈希 |
| `…/evaluation/workspace/artifacts/v2/` | 主模型、十模型指标、CV/OOF、70项LOBO、案例、字典、年龄/品牌边审计、图表与验证 |
| `artifacts/paired_revision/final-v4/` | 固定条件、共同划分、两策略完整模型/逐种子预测、配对区间、非空产物哈希 |
| `artifacts/final_v4_showcase.json` | 作者声明的非空文案、冻结模型结果、支持链接、storyboard及checkpoint哈希 |
| `artifacts/v2/` | 历史214条/171-43实验，追溯与旧策略输入，明确不是当前报告结果 |
| `data/youtube/` | 219条冻结平台请求与证据 |
| `output/pdf/Group6_Final_Report.pdf` | 当前四页报告；保留稳定文件名，作者已替换 |
| `output/latex/Group6_Final_Report.tex` | 与生成器同步的IEEE源码 |
| `src/make_ieee_report.py` | 只读取v4绑定结果生成报告，不硬编码历史结果 |

当前结果、模型和必要输入已随本分支上传。凭证、机器依赖、缓存、日志、旧托管网站和重复ZIP不上传。可见会话导出只留本地课程包，避免混入公开代码仓库。

## 11. 验证

```bash
python -m unittest discover -s Scripts -p 'test_*.py'
python src/validate_final_v4.py
python src/validate_report_format.py
python Scripts/validate_app_experiment.py --input artifacts/isolated_revision/final-v4 --check-app
python Scripts/verify_integration.py --check-app
```

检查覆盖：注册活动在外层与内层同组、13条隔离、标签来源、十模型定量测量、品牌均值/零品牌边、真实图来源、共享配对控制、产物哈希、201条新推理、214条历史推理、优化、作者Brief与年龄警告、实际版本切换、作者邮箱及4页A4。数值验证使用明确容差；AppTest不是人工浏览器排版验收。报告会另行渲染检查。

## 12. 重新训练与生成报告

冻结结果可直接使用。重新训练必须选不存在的新目录，不能覆盖final-v4。以下命令先只输出计划：

```bash
python Scripts/run_experiment.py --output artifacts/isolated_revision/reproduce-v4
```

显式训练（包括LOBO）：

```bash
python Scripts/run_experiment.py --output artifacts/isolated_revision/reproduce-v4 --run --acknowledge-exploratory
python Scripts/run_paired_experiment.py --input artifacts/isolated_revision/reproduce-v4 --output artifacts/paired_revision/reproduce-v4 --run --acknowledge-exploratory
```

参数表示明确接受仍未解决的创意源注释/活动谱系限制；不会重新抓取平台、推测标签或覆盖旧结果。完整输出通过manifest校验后才能出现在网页中。

本次v4图表/作者Brief/报告重建：

```bash
python src/finalize_v4.py
python src/make_ieee_report.py
TECTONIC_CACHE_DIR="$PWD/tmp/tectonic-cache" .project-deps/tex/tectonic --keep-logs -o output/pdf output/latex/Group6_Final_Report.tex
```

`finalize_v4.py` 不重训，只使用当前配置绑定的冻结模型生成定量训练表、条件区间、真实图和声明Brief；重建可能有平台浮点尾数差异。其他机器需自行安装Tectonic，替换本地helper路径。若要发布重新训练的另一版，先验证新完整目录，再更新 `artifacts/final_v4.json` 的三个实验路径并重建、核验报告；不要仅编辑哈希绕过检查。

## 13. 交付包与GitHub

```bash
python src/export_transcript.py  # 仅在具有本次真实会话日志的本机执行
python src/package_project.py
```

ZIP输出为 `output/Group6_Final_Project.zip`，包含Scripts运行依赖、v4冻结实验、配对结果、作者、网页、四页报告和SHA-256清单；缺少本机会话日志的克隆仍可打包代码/报告，不伪造会话。最终提交时另核对实际课程要求，未代为提交Blackboard或发送邮件。

公开仓库：[chengz1xuan/stat6207-group6-video-gnn](https://github.com/chengz1xuan/stat6207-group6-video-gnn)。本次提交仅推送新分支，不合并main或修改明坤原分支。

```bash
git clone --branch report/campaign-audit-v4 https://github.com/chengz1xuan/stat6207-group6-video-gnn.git
cd stat6207-group6-video-gnn
conda activate stat6207
python -m streamlit run app.py
```

## 14. IEEE排版与参考文献核验

摘要由IEEEtran原生abstract环境整段加粗；T1编码修复XeTeX下Times字体粗体/斜体回退的问题。PDF使用嵌入Times兼容字体，图表使用嵌入TrueType字体。四位作者按IEEE姓名—单位—地区—邮箱独立块排成两行，报告使用“名＋姓”，原始姓在前形式保留在AUTHORS.json的name字段，ieee_name保存报告显示形式。

参考文献已核验，详见[核验记录](references/REFERENCE_AUDIT.md)：GraphSAGE为NIPS2017真实论文，补齐1024–1034页；PyG补充真实ICLR2019 RLGM Workshop论文；算子引用改为具体SAGEConv官方API；IEEE模板明确属于格式来源。原IEEE网页自动抓取受robots限制，已对照IEEE会议托管副本和官方IEEEtran分发说明，未虚构研究文献或DOI。

`python src/validate_report_format.py`检查实际PDF字体粗体标志、作者/邮箱对应与两行布局、四条具体参考链接、字体嵌入及四页A4，而非只检查源码是否写了粗体命令。实验数据、模型和性能结果未因这次格式/引用修订而改变。
