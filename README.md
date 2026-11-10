# STAT6207 Group 6：基于 GNN 的广告视频播放量预测

本项目对应 **Topic 3：视频受众表现预测与创意设置优化**。使用 YouTube 广告视频的创意概念、文本证据和图结构，训练紧凑的图神经网络，比较不同节点、边及非图基线的预测效果，并通过 **Streamlit + Python** 展示实验结果和创意建议。

**2026-11-10 审计修正版：** 保诚周年活动人工统一后已重训；旧划分指标被替代，新增品牌均值与去品牌边对照，网页明确累计播放量与外推边界。

当前提交版本采用“**一个视频对应一个子图**”作为主要设计，以原始 `candidate_edges.csv` 中的关系为基础，对比加入语义相似边后的结果；“一个视频对应一个节点”作为扩展实验。所有主实验使用同一个按 campaign 分组的约 80/20 训练测试划分。

**日常查看项目只需启动 Streamlit，已有数据和模型可以直接使用。** 重新采集、重新训练、生成报告是独立操作，不是启动网页的前置步骤。

---

## 1. 阅读入口与最终交付物

| 目的 | 入口 | 说明 |
|---|---|---|
| 启动网页 | [app.py](app.py) | 本地 Streamlit 应用，直接调用 Python 模型 |
| 了解课程要求 | [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md) | 课程与 TA 补充要求、实验及提交说明 |
| 阅读最终报告 | [Group6_Final_Report.pdf](output/pdf/Group6_Final_Report.pdf) | **4 页 A4、IEEEtran 双栏**，包含图表和参考文献 |
| 修改报告 | [Group6_Final_Report.tex](output/latex/Group6_Final_Report.tex) | 可编辑 LaTeX 源码；数值由实验文件生成 |
| 理解数据字段 | [DATA_DICTIONARY.md](data/processed/DATA_DICTIONARY.md) | 原始标签位置、采集字段、缺失值、图与划分说明 |
| 查看训练和测试结果 | [artifacts/v2](artifacts/v2) | 当前正式实验的指标、预测、模型、案例与验证结果 |
| 查看实际对话 | `output/session_transcript.md`（仅本地/课程 ZIP） | 真实可见用户/助手消息，注明导出截止时点 |
| 查看工作过程摘要 | [SESSION_WORKLOG.md](SESSION_WORKLOG.md) | 工作记录，独立于实际对话文本 |
| 获取完整交付包 | `output/Group6_Final_Project.zip`（本地生成） | 包含应用、代码、数据、模型、报告及复现资料 |

当前报告和应用均以 `artifacts/v2/` 的结果为准。旧的 10 页报告位于 `output/archive/`，用于保留历史，不是提交报告。旧静态网站位于 `website/`，不作为当前应用入口，也不包含在当前交付包中。

## 2. 快速启动：已有数据与模型

### 2.1 在当前电脑运行

在项目根目录打开终端：

```bash
cd "/Users/chengzixuan/CUHK/STAT 6207/Final Project"
conda activate stat6207
python -m streamlit run app.py
```

浏览器打开 **http://localhost:8501**，或 **http://127.0.0.1:8501**。

终端中出现 Streamlit 地址后，保持该终端运行。结束服务时，在对应终端按 `Ctrl+C`。关闭浏览器页面不会自动结束 Python 服务。

若终端尚未配置 `conda activate`，当前电脑也可以直接指定已有环境的解释器：

```bash
/opt/anaconda3/envs/stat6207/bin/python -m streamlit run app.py
```

启动前可以核对解释器，避免误用系统 Python 或其他 Conda 环境：

```bash
python -c "import sys; print(sys.executable)"
```

当前电脑预期输出：

```text
/opt/anaconda3/envs/stat6207/bin/python
```

### 2.2 本地配置的作用范围

[.streamlit/config.toml](.streamlit/config.toml) 只配置当前项目：

- 服务监听 `127.0.0.1`，供本机访问。
- 使用项目配色，关闭使用统计上报。
- 采用 headless 模式，由用户自行打开浏览器。
- 关闭文件监控；修改应用代码后，建议停止服务再重新启动。

这些配置不会修改电脑的全局 Python 默认值或 Conda 默认环境。当前实现不需要登录托管网站，也不需要 Node.js 来运行预测。

`localhost` 指访问者自己的电脑。将本地链接发给其他人，不会使对方连接到你的应用；对方可以下载项目并按相同步骤启动。

## 3. Streamlit 网页使用指南

应用顶部显示样本量、品牌数、训练测试数量和主模型 Log RMSE。左侧栏提供实验摘要与最终 PDF 下载。页面分为五个标签页。

### 3.1 研究结果

用于查看和解释实验，而非修改训练模型：

1. 查看十种基线/图模型在同一测试集上的 Log MAE、Log RMSE、Spearman 和原始播放量 MAE。
2. 查看误差图，以及按 campaign 重采样的 bootstrap 区间。
3. 查看 leave-one-brand-out 的品牌宏平均结果。
4. 查看不同品牌、播放量区间的误差切片及改进方向。

主模型来自训练集交叉验证选择。测试集中某个扩展模型分数更好，不会反过来用于重新选择主模型。

### 3.2 图结构浏览

用于检查一个视频实际对应的 GNN 子图：

1. 按品牌筛选，再选择视频；选项标有 `train` 或 `test`。
2. 查看视频根节点、创意概念节点、文本证据节点。
3. 切换“显示新增的语义相似边”，对比原始结构与相似边扩展。
4. 使用“检查节点”选择框读取完整文本、节点 ID、关联边及原始来源边 ID。
5. 查看原始观测播放量、采集时间，并打开来源 YouTube 链接。

颜色含义：深绿为视频根节点，浅黄为创意概念，浅绿为文本证据，橙色虚线为新增启发式相似边。

**图页面的开关只影响显示。** 预测页始终使用已训练主模型对应的边定义；切换显示不会将模型变成另一个消融模型。

### 3.3 预测与优化建议

支持两种输入方式：

- **新视频 Brief**：以训练集中的产品语境作为初始表单，修改为自己的创意设置与文案。新视频图仅由声明输入构建，不借用其他视频的抽取证据节点。
- **已有视频**：载入一个参考视频的设置及图结构，再编辑创意字段。原抽取证据保持固定，因此需要人工检查修改后的分类、文案与证据是否一致。

可编辑的主要输入包括品牌、creative mechanic、structure、evidence mode、CTA explicitness、proof strength、pacing、production complexity、时长、假定曝光天数，以及 message、hook、value promise、CTA 文本。

填写后选择：

| 操作 | 返回内容 |
|---|---|
| 预测播放量 | 由本地 GNN 计算的累计播放量、log 预测及残差敏感度范围 |
| 搜索支持的创意设置 | 原设置预测、支持范围内的最佳预测、建议修改、候选数量与训练集来源 |
| 下载本次预测 JSON | 保存最近一次提交的输入和结果，便于讨论与记录 |

更改输入后需要再次提交表单，才会计算新结果。**目标是采集时累计播放量，曝光年龄只是输入协变量，不是经过验证的未来预测窗口。** 修正后训练集年龄为 36.5–5620.5 天，中位数 718.5 天；新 Brief 默认使用该中位数。输入 30 天会在预测结果及 JSON 中标为范围外推，不能宣称“上线 30 天可获得的播放量”。年龄处于范围内，也不代表新 Brief 已被独立验证。

自动搜索只修改 `creative_mechanic`、`structure_type`、`pacing`，最多同时改变两项。候选三元组必须在训练集中出现至少两次；若没有候选提高预测，则保留原设置。

### 3.4 案例与 Storyboard

包含一个时长 30 秒、曝光年龄取训练中位数 718.5 天的新视频 storyboard，以及六个固定测试视频案例。30 秒是片长；本项目没有验证固定 30 天播放量预测。

每个已有视频案例包含：目标、原始创意元素、原视频实测播放量、原设置预测、建议设置预测、残差范围、修改建议、训练来源和可迁移启示。

新视频及修改后的视频没有实测响应。“建议设置预测”不能与“原视频实测”一起被解释成已经发生的提升。

### 3.5 数据与报告

可以查看冻结指标表、单条采集证据、HTTP/抽取状态、字段来源与时间戳，并下载 CSV、数据字典和实验指标。最终四页报告通过左侧栏下载。

## 4. 环境与依赖

### 4.1 当前电脑

当前项目使用已有 Conda 环境 `stat6207`，Python 版本为 **3.11.16**。Streamlit **1.66.1** 已安装在该环境中。

核心计算依赖包括 NumPy、Pandas、PyTorch、scikit-learn、SciPy 和 Matplotlib。PyTorch Geometric、PDF 工具及排版工具等辅助依赖位于项目内 `.project-deps/`；代码会为相关模块添加该搜索路径。

[requirements.txt](requirements.txt) 记录本次使用的主要 Python 包版本。`.project-deps/`、`.venv/` 和 TeX 下载缓存不包含在交付包中。

### 4.2 在其他电脑配置

如果没有 `stat6207` 环境，可新建一个独立环境；已有同名环境时直接激活，不必重复创建：

```bash
conda create -n stat6207 python=3.11
conda activate stat6207
python -m pip install -r requirements.txt
```

随后在项目根目录运行：

```bash
python -m streamlit run app.py
```

`requirements.txt` 是已验证运行环境的版本记录。其他操作系统、CPU 架构或软件源可能无法提供完全相同的构建；若需要调整版本，应重新运行项目验证，而不是默认结果完全一致。

启动应用不需要 LaTeX。只有重新编译报告时才需要 Tectonic 或已有的 LaTeX 安装。

### 4.3 模型文件加载

应用通过 `src/model_service.py` 加载 `artifacts/v2/primary_gnn.pt`，使用 CPU 直接执行 PyTorch Geometric 推理，不会在启动时重训。

checkpoint 包含权重、训练得到的预处理器、模型配置和训练/测试 ID。`src/checkpoint_pickle.py` 处理训练脚本中的类名，避免依赖 Streamlit 的 `__main__` 命名空间。该加载方式仅用于项目中可信的本地 checkpoint。

## 5. 项目目录与文件分工

```text
Final Project/
├── app.py                         # 当前 Streamlit 应用入口
├── .streamlit/config.toml          # 当前文件夹的应用配置
├── README.md                      # 使用、复现与结果说明
├── PROJECT_REQUIREMENTS.md        # 课程/TA 要求对应说明
├── requirements.txt               # Python 包版本记录
├── SESSION_WORKLOG.md             # 工作过程摘要
├── data_topic3/                   # 原始品牌包，保留来源文件
├── data/
│   ├── youtube/                   # 冻结采集 CSV、逐视频 JSON、采集摘要
│   └── processed/                 # 统一表、当前建模表、标签来源与数据字典
├── artifacts/
│   └── v2/                        # 当前正式实验结果、图、checkpoint 与验证
├── src/
│   ├── revision.py                # 正式训练/评估/优化流程
│   ├── model_service.py           # 应用使用的直接 Python 推理与设置搜索
│   ├── checkpoint_pickle.py       # checkpoint 类名兼容加载
│   ├── collect_youtube.py         # 公开 YouTube 元数据采集
│   ├── validate_revision.py        # 数据划分、来源、泄漏与实验检查
│   ├── test_streamlit_app.py       # 数值回归与 Streamlit 表单测试
│   ├── training_measurements.py    # 汇总各模型训练目标和定量测量
│   ├── make_ieee_report.py        # 从实验结果生成 IEEE 报告源码
│   ├── setup_ieee.py              # 获取 IEEEtran 和本地排版工具
│   ├── export_transcript.py       # 从本次本地会话日志导出可见消息
│   └── package_project.py         # 生成交付包及 SHA-256 文件清单
├── references/                    # 原始计划及 IEEE 模板资料
├── output/
│   ├── pdf/Group6_Final_Report.pdf # 最终四页提交报告
│   ├── latex/                     # 报告源码、类文件和图表
│   ├── session_transcript.md      # 实际对话导出
│   ├── SHA256SUMS.json            # 交付内容的文件哈希
│   └── Group6_Final_Project.zip   # 完整交付包
└── tmp/                           # 渲染图片、字体与排版缓存等中间文件
```

`src/pipeline.py` 还提供当前流程引用的 `Features` 等工具类，但其直接执行入口属于旧版实验。正式训练入口是 **`src/revision.py`**。旧版 `make_report.py`、`validate.py`、静态网页辅助脚本等不用于当前正式结果。

`artifacts/` 根目录及 `artifacts/v1/` 可能保留旧版结果。对照报告时，请明确进入 `artifacts/v2/`，避免混用不同数据快照和划分的指标。

## 6. 数据来源、标签与冻结策略

### 6.1 原始库存

按每个品牌的最新版本统一保留，得到 **219 个唯一视频、14 个品牌**。HSBC 和 Hang Seng 使用 v0.2；MTR 文件夹名称虽含 15，但实际视频表有 14 条记录。

原始包主要包含：

| 文件 | 用途 |
|---|---|
| `video_nodes.csv` | 视频 ID、标题、品牌、URL、日期、时长及创意设置 |
| `aspect_nodes.csv` | 创意概念及其来源 |
| `evidence_nodes.csv` | 元数据、文本证据、受众指标及质量信息 |
| `candidate_edges.csv` | 原始关系、候选边及说明 |
| 数据字典/摘要文件 | 包内字段说明、审计与限制 |

原始受众指标通常位于 `evidence_nodes.csv`：筛选 `kind == engagement_stats`，JSON 解析 `value_json`，若不存在则解析 `text`，读取 `views`、`likes`、`comments` 等键。不同包的 `video_id` 可能带 `video:` 前缀，统一后连接。McDonald's 的视频表还直接提供相关列。

原始覆盖为 **219 条 views、184 条非缺失 likes、0 条非缺失 comments**。这里“0 条非缺失 comments”表示没有可用评论计数，不表示视频没有评论。

### 6.2 本次 YouTube 采集

冻结采集窗口为 **2026-11-09 12:01:03–12:02:18 UTC**，即香港时间 **20:01:03–20:02:18**。

| 采集项目 | 覆盖情况 |
|---|---:|
| 尝试的原始视频 URL | 219 |
| 精确播放量 | 214 |
| 精确点赞数 | 175 |
| 频道身份 | 214 |
| 初始公开元数据中的精确评论计数 | 未取得 |

采集读取公开 watch 页面的当前视频元数据，记录 URL、UTC 请求时间、HTTP/抽取结果、响应 SHA-256、原始计数片段及字段路径。每个视频的 JSON 证据保存在 `data/youtube/<youtube_id>.json`。

五个无法取得新精确播放量的视频从新版建模样本中排除；没有用历史计数补入，也没有从创意评分估算标签。原始 219 条历史播放量另行保留用于追溯。

频道订阅数的缩写显示仅作描述，不能当作精确计数。当前采集没有取得新确认的上传日期，曝光时长仍依据原始库存中的发布日期计算，这也是分析限制之一。

### 6.3 关键数据表

| 文件 | 内容 |
|---|---|
| `data/processed/videos.csv` | 统一后的原始库存及历史标签 |
| `data/processed/label_provenance.csv` | 历史标签来源、证据 ID、可用时间戳 |
| `data/youtube/snapshot.csv` | 本次 219 个链接的冻结采集结果 |
| `data/youtube/summary.json` | 覆盖率、采集时间与策略摘要 |
| `data/processed/videos_v2.csv` | 214 条新标签构成的正式建模表 |

逐字段说明见 [数据字典](data/processed/DATA_DICTIONARY.md)。

### 6.4 为什么不在启动网页时重新采集

网页展示的是一个已经冻结并完成评估的实验。实时重抓会改变目标标签、样本可用性和曝光时长，导致网页与报告不再对应。

`src/collect_youtube.py` 默认复用已有的逐视频 JSON，所以再次执行并不等同于采集一个全新时点。若确需新快照，应先复制项目或备份结果，再在采集脚本中设置独立的输出目录，并同步调整训练输入；当前脚本没有提供独立输出目录的命令行参数。

新快照必须作为新实验重新训练、验证、生成报告，不能只改 CSV 后继续沿用旧指标。

## 7. 图定义与模型

### 7.1 主要设计：一个视频是一个子图

每个子图包含视频/上下文根节点、创意概念节点和经过筛选的文本证据节点。214 个图合计 **6,684 个节点、7,780 条保留的原始关系**。

核心关系直接来自 `candidate_edges.csv`：

| 关系 | 含义 |
|---|---|
| `video_has_aspect` | 视频包含某个创意概念 |
| `video_has_evidence` | 视频对应某条文本证据 |
| `aspect_supported_by_evidence` | 概念与支持它的证据连接 |

来源包、原始边 ID 和启发式标记被保留。原始支持边可能也是启发式关系，并非已经验证的因果关系。用于消息传递时，将保留的关系对称化。

节点字段使用固定允许列表，去重并限制重复概念/证据数量。播放量、点赞、评论、`fit_score`、风险评估及抽取质量分数等不进入预测图；文本还会移除包含指定受众响应词的句子。该过滤是可审计规则，不能保证消除所有间接信息或标注偏差。

### 7.2 结构＋相似边扩展

在相同节点信息和模型结构上，添加视频内部非根节点之间的语义相似边：

- 文本编码器只在相应训练数据上拟合。
- 使用 TF-IDF/SVD 文本表示的 cosine similarity。
- 相似度至少为 `0.55`，每个节点最多选择两个邻居。
- 添加后对称化、去重，并标记为启发式相似边。

结构版与相似边扩展版使用相同主实验划分，形成直接边消融。

### 7.3 视频节点扩展

每个视频对应一个节点。经审计，原始同品牌、作者及创意类别关系合并成二元邻接后，**1,558 对结构连接恰好构成完整品牌内连接图**。创意/作者关系标签未增加独立邻接信息，因此该模型的表现不能证明创意概念关系有效。相似边扩展版再增加五个训练特征近邻。

新增 `Brand mean`：只用训练集各品牌的平均 log 播放量预测；未见品牌退回全局训练 log 均值，绝不使用测试标签。新增 `Node no-brand edges`：移除所有同品牌端点连接，包括被作者/创意等其他标签重复支持的连接，故剩零条边；保留品牌输入特征、架构和相同调参协议。该对照隔离邻接作用，不是移除全部品牌信息。

验证/测试视频仅从训练参考节点接收消息，不向训练节点发送消息，也不在不同测试节点之间传播消息。这避免将整张测试图无条件纳入训练参考。

### 7.4 输入与网络结构

根节点输入包括品牌与七个创意类别、log 时长、log 曝光时长、缺失指示及创意文本表示。其他节点包括节点类型、字段类型和文本表示。

主要 GNN 使用 PyTorch Geometric 的 mean `SAGEConv`，两层各 24 个隐藏单元。拼接根节点表示与全图平均池化表示后，以线性输出头预测标准化的 log 播放量。

对照包括训练集 log 均值预测器、Metadata Ridge、Pooled Ridge、无消息传递的 Pooled MLP。MLP 保留相同节点信息与读出方式，但参数量并不与 GNN 完全一致。

## 8. 训练、划分与评估协议

### 8.1 定量目标

```text
y = ln(1 + views)
z = (y - mean_train) / std_train
神经网络训练损失 = MSE(z_pred, z)
```

报告 Log MAE、Log RMSE、Spearman 排序相关，以及反变换后的原始播放量误差。MAE/RMSE 越小越好；Spearman 越高表示排序更一致。常数均值预测器的 Spearman 不定义。

`exp(y_pred) - 1` 是 log 尺度点预测的反变换，不是无偏的算术期望播放量。Log RMSE 也不能直接解释成“错误了多少次播放”。

### 8.2 按 campaign 分组的 80/20 划分

原始规则合并同品牌规范化 campaign 名称与完全相同的 main message。人工复核发现保诚 170 周年活动的中英文名称被误分组，现通过 `data/processed/campaign_overrides.csv` 将以下视频统一为 `prudential_170th_anniversary_2019`，并保留既有分组的传递合并关系：

- `cmvbu5bpx004mbnuobvnt3njr` / YouTube `jlB5oiVibQU`
- `cmvbu6xs9004wbnuo3n7cfzh3` / YouTube `m8qVqW63VQs`

修正后 214 条样本有 **198 个分组**。沿用“从 seed 42 起选择第一个产生 round(20% × N) 条测试视频的 seed”的规则，只看组大小，不看标签或得分，现得到 **seed 50、171 训练 / 43 测试**（79.91% / 20.09%）。两条周年活动视频均在训练集，且在训练内部各 CV 折保持同组。测试集包含 40 个活动组。分组和 YouTube ID 不跨边界；其他双语别名仍可能需要进一步复核。

**这是检查过旧测试误差后修正的内部评估，不能包装为全新未接触的外部测试集。** 冻结标签不变，旧结果在 `artifacts/v2/correction_history.json` 中留档，当前指标全部来自修正后的重新训练。

### 8.3 训练内部调参及最终重训

1. 在外层 171 条训练样本中做三折 `GroupKFold`。
2. 每折独立拟合类别编码、插补、缩放、词表、SVD 和训练参考图。
3. Ridge 搜索 `alpha ∈ {10, 100}`；神经网络搜索 `weight_decay ∈ {0.01, 0.1}`。
4. 使用 Adam，学习率 `0.01`，最多 260 轮，内部验证早停 patience 35。
5. 按平均折内 Log RMSE 选择配置，并取所选配置的折内最佳轮数中位数。
6. 在完整 171 条训练数据上重新拟合预处理和模型，再评估 43 条测试数据。

测试标签不用于调参、停止轮次、模型选择或图规则选择。内部早停分数用于调参，不作为无偏的最终泛化估计。

### 8.4 不确定性与跨品牌压力测试

- **Campaign bootstrap**：在测试 campaign 层面重采样 2,000 次，seed 44；模型间使用相同重采样以计算配对差异。区间以当前划分和拟合模型为条件，不包含划分与模型选择的不确定性。
- **随机初始化敏感度**：额外使用 seed 7、21、84，不根据测试结果替换主 seed 42。
- **Leave-one-brand-out**：逐一完全留出 14 个品牌，对其余品牌重新拟合并内部选择停止轮次。比较五种模型（含未见品牌退回全局均值的 Brand mean），产生 70 个模型－品牌评估。该压力测试固定正则化配置，与主 80/20 协议分开报告。
- **案例中的残差范围**：来自训练折外绝对 log 残差的名义 90% 范围，仅作敏感度参考，不保证个体预测覆盖率。

## 9. 当前结果与正确解读

### 9.1 同一主测试集的比较

| 模型 | Test Log MAE | Test Log RMSE | Spearman |
|---|---:|---:|---:|
| Mean | 2.282 | 2.646 | 不定义 |
| Brand mean | 1.989 | 2.622 | 0.281 |
| Metadata Ridge | 1.989 | 2.375 | 0.408 |
| Pooled Ridge | 1.904 | 2.345 | 0.450 |
| Pooled MLP | 1.827 | 2.360 | 0.436 |
| **Subgraph structural** | 1.795 | 2.402 | 0.447 |
| Subgraph + similarity | 1.917 | 2.286 | 0.490 |
| Node structural | 1.952 | 2.344 | 0.550 |
| Node + similarity | 1.886 | 2.369 | 0.414 |
| Node no-brand edges | 1.948 | 2.382 | 0.362 |

主子图由训练 CV 选择为 **Subgraph structural**；完整比较套件的 CV 最优为 **Node no-brand edges**。测试表现不用于改选主模型。主子图 Log RMSE 2.402，高于 Metadata Ridge 的 2.375、Pooled Ridge 的 2.345 和视频节点模型的 2.344，**不能声称概念子图已优于简单基线**。相似边子图虽然测试 RMSE 较低，也不能看过测试结果后替换预先选择的主模型。

主子图 campaign-bootstrap 95% Log RMSE 区间为 [1.759, 2.939]。品牌连接模型相对去品牌边模型的配对 RMSE 差异区间为 [-0.215, 0.142]，包含零；其邻接增益不稳定，且原结构图仅编码品牌分组，不能据此证明创意关系有效。完整配对结果见 `paired_comparisons.json`。

### 9.2 跨品牌结果

| 模型 | LOBO Macro Log MAE | LOBO Macro Log RMSE | Macro Spearman |
|---|---:|---:|---:|
| Pooled Ridge | 2.145 | 2.563 | 0.262 |
| Pooled MLP | 2.188 | 2.634 | 0.237 |
| Subgraph structural | 2.134 | 2.593 | 0.212 |
| Subgraph + similarity | 2.136 | 2.589 | 0.174 |
| Brand mean | 2.261 | 2.672 | 不定义 |

跨品牌误差仍较大。Brand mean 在未知品牌上退回全局训练 log 均值，因此品牌内排序相关不定义。这个小样本实验没有证明概念关系有额外预测价值，也没有验证新视频的固定时间窗口流量承诺。

### 9.2.1 两类最大误差应分开讨论

修正前最大 log 误差：麦当劳旧广告 `[香港經典廣告](1996)麥當勞 金鼠獻瑞`，YouTube `P25f_dDd2MI`，实测 **1,019**，原预测约 **442,536**。采集到的频道为 `TPB HK CM 香港電視廣告庫 (香港電視廣告庫)`，属于第三方广告转载。1996 是广告创意年份；供给表的上传日期是 2026-04-28，对应采集年龄约 195.5 天。官方品牌概念无法保证转载频道有相同分发、受众或投放曝光，这些是潜在解释，未被识别为因果效应。

该视频在修正划分中进入训练集，修正后的预测只能作为样本内诊断，不能再称为测试误差。历史失败保存在 `reupload_error_audit.json`。当前最大测试 log 误差为 IKEA `太古猩喺度！（都足本短版）`：实测 **958**，预测约 **839,933**；说明官方频道身份本身也不能消除曝光和分发差异。

报告中的概念图是实际训练图的可读诱导子图，原始节点/边及选图规则在 `report_graph.json`；散点图包含全部 43 条修正后测试预测，不混入训练预测。

### 9.3 创意建议的解释边界

六个已有视频案例由固定测试索引确定，而非挑选预测提升最大的案例。来源实例仅来自训练集。原始观测播放量属于原视频；修改后视频和新 storyboard 没有观测结果。

推荐应表述为“在模型与当前样本中关联较强的设置”或“值得做控制实验的假设”。未测量品牌认知、销售额，也没有通过随机曝光实验识别因果效果。媒体预算、频道受众、人物吸引力、时间窗口和抽取偏差都可能解释观测差异。

## 10. 结果文件速查

以下路径均位于 `artifacts/v2/`：

| 文件 | 应当用来回答的问题 |
|---|---|
| `audit.json` | 最终样本量、缺失覆盖、排除视频有哪些？ |
| `splits.csv` / `split_policy.json` | 哪些视频属于训练或测试？分组与比例是什么？ |
| `graph_audit.json` | 节点、关系数量与过滤规则是什么？ |
| `raw_graphs.json` | 每个视频的原始建模节点与结构边是什么？ |
| `edge_provenance.csv` | 某条边来自哪个包和哪个原始边 ID？ |
| `cross_validation.json` | 参数如何选择？每折误差和停止轮次是多少？ |
| `training_histories.json` | 最终神经网络的训练损失如何变化？ |
| `training_measurements.csv` | 各模型目标、训练误差及定量测量是什么？ |
| `metrics.json` | 十种模型的测试指标、主模型及配置是什么？ |
| `test_predictions.csv` | 每条测试视频的预测与实测分别是多少？ |
| `oof_predictions.csv` | 训练折外预测是什么？ |
| `uncertainty.json` | Bootstrap 区间及相似边配对差异是什么？ |
| `seed_sensitivity.json` | 换初始化后结果如何？ |
| `leave_brand_out.csv` / `leave_brand_out_summary.json` | 各品牌/宏平均的迁移误差是什么？ |
| `campaign_overrides.csv`（data/processed） | 哪些中英文活动别名经过人工统一？ |
| `inner_folds.csv` | 人工合并活动是否在训练内部各折保持同组？ |
| `brand_edge_audit.json` / `video_edge_provenance.csv` | 品牌邻接是否主导视频节点图？去边后剩多少？ |
| `age_support.json` | 训练年龄范围和新 Brief 默认值如何确定？ |
| `correction_history.json` / `reupload_error_audit.json` | 哪些旧结果被替代？转载失败与当前测试如何区分？ |
| `paired_comparisons.json` / `report_graph.json` | 对照差异的不确定性及真实图来源是什么？ |
| `error_slices.json` | 哪些品牌、播放量范围容易出错？ |
| `optimization.json` | 六个案例、新视频 storyboard 与搜索限制是什么？ |
| `primary_gnn.pt` | Streamlit 实际加载的主模型是什么？ |
| `web_results.json` | 应用读取的结果、视频及图数据汇总是什么？ |
| `run_manifest.json` | 运行版本、训练目标、代码与快照哈希是什么？ |
| `validation.json` | 划分、标签来源、图与泄漏检查是否通过？ |
| `streamlit_validation.json` | 数值一致性及表单交互检查是否通过？ |
| `streamlit_browser_validation.json` | 真实浏览器中的图渲染、优化和布局检查是否通过？ |

`web_results.json` 保留历史文件名，目前是本地 Streamlit 读取的数据文件，并不要求网页托管服务。

## 11. 复现与维护操作

以下操作应从项目根目录、在 `stat6207` 中执行。重新训练会写入正式结果文件；如需保留当前提交版本，请先复制项目或备份 `artifacts/v2/`、`data/processed/` 和 `output/`。

### 11.1 只检查当前交付结果

```bash
conda activate stat6207
python src/validate_revision.py
python src/test_streamlit_app.py
```

第一条同时检查标签来源、外层和内层分组隔离、保诚别名、训练品牌均值/未知品牌回退、品牌连接来源与零边对照、年龄支持范围、历史/当前误差隔离及真实图节点边。第二条对照全部 214 条预测和六个优化案例，并用 Streamlit AppTest 检查新视频、已有视频、编辑设置、图开关、年龄中位数默认值及 30 天外推提示。

验证结果会写入对应 JSON；无需为此重新抓取 YouTube 或重新训练。

### 11.2 从冻结数据重新训练

```bash
python src/revision.py
python src/training_measurements.py
python src/validate_revision.py
python src/test_streamlit_app.py
```

正式训练脚本执行数据准备、图构造、80/20 分组划分、训练内部三折调参、最终重训、测试评估、bootstrap、随机种子敏感度、跨品牌压力测试及案例生成。

它使用已有的 `data/processed/videos.csv`、原始品牌文件和冻结 `data/youtube/snapshot.csv`，不会自动重新抓取线上数据。运行时间依硬件及软件版本而变化。

完成后重启 Streamlit，使模型缓存与结果文件保持一致。若改了特征、图定义或预处理类，需要重新训练和重新验证；不能只把新代码配上旧 checkpoint 就假定兼容。

### 11.3 重新生成四页 IEEE 报告

先从最新结果生成 LaTeX：

```bash
python src/make_ieee_report.py
```

在当前电脑已有项目本地 Tectonic 时编译：

```bash
TECTONIC_CACHE_DIR="$PWD/tmp/tectonic-cache" .project-deps/tex/tectonic --keep-logs -o output/pdf output/latex/Group6_Final_Report.tex
```

如尚未获取排版工具，可执行：

```bash
python src/setup_ieee.py
```

该安装辅助脚本获取 IEEEtran 分发文件和 **macOS Apple Silicon** 版本的 Tectonic，需联网，不是跨平台安装器。已有其他 LaTeX 环境时，可进入 `output/latex/` 使用 `pdflatex` 或 `xelatex` 编译，并保留 `a4paper,conference` 类选项。

生成后必须检查：页数不超过四页、页面为 A4、图表和参考文献完整、没有溢出或遮挡。数值变动可能改变版面，不能仅因为编译成功就认定满足页数限制。

应用下载按钮直接读取 `output/pdf/Group6_Final_Report.pdf`，不需要把报告复制到静态网页目录。

### 11.4 导出对话与重新打包

```bash
python src/export_transcript.py
python src/package_project.py
```

对话导出依赖当前电脑上本次会话的本地日志。在其他电脑没有这些日志时，保留已交付的 transcript，或从原聊天工具导出实际会话；不能用工作摘要代替真实对话。

打包脚本更新 `output/Group6_Final_Project.zip` 和 `output/SHA256SUMS.json`，包含当前 README、Streamlit 应用、模型、数据、报告及源码，排除旧静态网站、旧版结果、环境依赖和中间文件。

## 12. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `conda: command not found` / 无法 activate | 当前终端没有加载 Conda。当前电脑可使用第 2 节中的完整解释器路径启动。 |
| `No module named streamlit` | 先核对 `sys.executable`，确保使用 `stat6207`；在新环境中按 `requirements.txt` 安装。 |
| `No module named torch_geometric` 或 `xxhash` | 当前电脑检查 `.project-deps/` 是否完整；其他电脑在所激活环境中安装项目依赖。 |
| 找不到 checkpoint、JSON 或 CSV | 检查是否完整解压项目，并保留目录结构；不要只复制 `app.py`。 |
| 8501 端口已占用 | 先检查是否已有应用在运行，或使用 `python -m streamlit run app.py --server.port 8502`，随后访问新端口。 |
| 浏览器打不开网页 | 确认终端进程仍在运行，打开终端实际打印的地址；本地链接不能作为公开分享链接。 |
| 修改 Python 代码后页面未变化 | 项目关闭了文件监控，停止并重启 Streamlit。 |
| 换模型后仍像旧结果 | 重启服务；应用缓存模型资源，模型、预处理器与结果文件也必须属于同一次实验。 |
| 图里文字太小 | 使用图的展开/缩放功能，并通过“检查节点”读取完整文本与来源 ID。 |
| 关闭相似边后预测未改变 | 图开关只改变显示，预测仍使用已训练主模型的边定义。 |
| 修改表单但预测未改变 | 点击预测或搜索按钮重新提交；页面显示的是最近一次提交的结果。 |
| 预测与观测差别很大，范围也很宽 | 当前小样本、遗漏变量和分布差异导致不确定性较大，应结合基线和误差分析解读。 |
| 评论数为空 | 本次初始公开元数据未取得精确评论数，不能补成 0。 |
| 搜索没有建议修改 | 在受约束候选中没有更高预测时，系统保留原设置，这是有效结果。 |
| `__main__.GraphFeatures` 或 `Features` 加载异常 | 使用 `model_service.py` 中的兼容加载方式；不要脱离类定义直接加载脚本生成的 checkpoint。 |
| LaTeX 工具不存在或首次编译缺宏包 | 启动网页不需要 LaTeX；重新生成报告时再安装适合当前平台的工具，首次编译可能需要联网。 |

## 13. 验证记录与提交说明

当前保存的验证记录包括：

- 全部 214 条 Python 预测与冻结输出一致，最大 log 差异为 0。
- 六个优化案例的结果与候选数量一致。
- 新 Brief 图不借用已有视频的证据节点。
- Streamlit 初始化、预测、优化、已有视频数值一致性、编辑提交与图开关均通过，应用异常为 0。
- 实际浏览器验证了图 SVG 渲染、优化表单、1440×1000 桌面与 390×844 手机布局，没有页面级横向溢出或 JavaScript 异常。
- 最终 PDF 为四页 A4，已渲染检查，没有 overfull 排版溢出。

这些是当前保存版本的验证记录。变更模型、依赖、数据或界面后，应重新运行受影响的检查。

课程要求于 **2026 年 11 月 12 日**前提交 PDF 报告及 session transcript。正式 PDF 的四页限制包含正文、图表和参考文献；详细数据、完整结果与代码单独提供。

提交时使用：

1. `output/pdf/Group6_Final_Report.pdf`；
2. `output/session_transcript.md`，注意其中注明的导出截止时点；
3. 按课程要求附上代码/数据/网页复现材料，完整项目包位于 `output/Group6_Final_Project.zip`。

本项目未自动提交 Blackboard，也未向 TA 发送邮件。`SESSION_WORKLOG.md` 是工作摘要，不替代实际对话文本。


## 14. GitHub 仓库与本地交付包

GitHub 版本以当前 Streamlit/Python 实现为入口，仓库为 [chengz1xuan/stat6207-group6-video-gnn](https://github.com/chengz1xuan/stat6207-group6-video-gnn)，按项目所有者授权保持公开可见。代码、冻结数据、训练模型、正式实验结果、IEEE 报告与源码用于协作和复现。

`.gitignore` 排除本地 Python 依赖、编辑器专用路径、凭证文件、缓存、日志、重复 ZIP、旧静态网站、旧实验及机器相关部署信息。个人会话文本 `output/session_transcript.md` 和本地交付说明保留在本机及课程交付包中，不随 Git 仓库上传。

克隆仓库后，按本 README 的环境配置和 Streamlit 启动步骤运行；无需访问旧托管站点。Git 仓库与课程 ZIP 包的内容范围有所区别，课程提交仍以第 13 节为准。


组员可以克隆公开仓库：

```bash
git clone https://github.com/chengz1xuan/stat6207-group6-video-gnn.git
cd stat6207-group6-video-gnn
conda activate stat6207
python -m streamlit run app.py
```
