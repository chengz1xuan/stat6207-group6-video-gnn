# 整合候选版使用说明

## 版本与边界

本目录以 `stat6207-group6-video-gnn-main-new.zip` 为基线，接入明坤分支的
`Scripts/`、实验发现及网页切换功能。两份 ZIP 没有 `.git` 历史，因此这是一份
内容整合候选版，不能证明远程分支必然 fast-forward 或零冲突。

main 的 `src/revision.py`、正式数据、模型、PDF、LaTeX 和生成报告脚本均保留。
明坤原分支的旧 `artifacts/v2`、旧报告和旧 `revision.py` 不覆盖 main。

## 已调整

- 网页支持选择通过校验的附加实验，动态显示样本量、模型、指标及下载文件。
- 保留 main 的品牌均值、去品牌边对照、累计播放量解释、年龄范围与外推提示。
- 实验切换清除旧预测及编辑状态；新 Brief 只传声明字段，不借用参考视频标签。
- 缺少附加实验时仍可使用 main；损坏的可选实验会显示警告并从列表排除。
  手动加载损坏实验仍然报错，不关闭模型、数据或训练源码的哈希校验。
- 推理年龄统计从所选模型的训练视频计算，兼容没有 age_support 的探索性导出。
- 新隔离评估调用最新 main 的训练代码，计划清单为十模型，不沿用旧八模型结论。

## Mac 本地验证

在本项目根目录，使用你已安装项目依赖的 Python 环境：

```bash
python -m unittest discover -s Scripts -p 'test_*.py'
python Scripts/verify_integration.py --check-app
python -m streamlit run app.py
```

如果使用项目虚拟环境，将 `python` 替换为 `.venv/bin/python`。
Windows 对应 `.venv\Scripts\python.exe`。不要复制 README 中其他同学的个人绝对路径。
`requirements.txt` 保留 main 的原环境记录；`Scripts/requirements-experiment.txt` 是较宽的
实验依赖范围，不是保证所有平台都经过验证的锁文件。测试时记录实际 Python 和包版本。

网页手动检查：原版结果、图结构、已有视频预测、新 Brief、设置优化、JSON 下载、
PDF 下载；拿到完整附加实验后再检查切换与返回，确认样本数、下载指标和表单状态对应。

## 缺失的明坤实验

两份输入 ZIP 均无 `artifacts/isolated_revision/full-app-v3/` 完整产物。
`Scripts/README.md` 描述的 161/40/13 和相应指标属于同学本机实验说明，本整合不把它们
当成已验证结果，也不据此改正式报告。请向同学取得整个目录及实际环境版本。

旧 bundle 可能因其训练源码与最新 main 不同而被拒绝。这是有意保留的防错检查，
不要编辑哈希来强制通过。若源码不匹配，应在整合代码上生成一个新的独立实验：

```bash
python Scripts/run_experiment.py --output artifacts/isolated_revision/integration-v1
```

上述命令只显示计划。如确实需要重新训练，再显式执行：

```bash
python Scripts/run_experiment.py --output artifacts/isolated_revision/integration-v1 --run --acknowledge-exploratory
python Scripts/validate_app_experiment.py --input artifacts/isolated_revision/integration-v1 --check-app
```

输出目录必须尚不存在。新的十模型结果必须重新测得，不能复用旧八模型数值。
本次整合未启动该全量训练。实验进入最终四页报告前，需补齐产物与验证；PDF 和
`src/make_ieee_report.py` 必须同步修改，否则重新生成会丢失手工补写。

## 如何交回 GitHub

优先从最新 main 建立 `integration/mingkun-compatible` 候选分支，然后应用提供的
`main-compatible.patch`。在远程 main 与输入 ZIP 一致时，`git apply --check` 应通过。

```bash
git switch -c integration/mingkun-compatible
git apply --check /path/to/main-compatible.patch
git apply /path/to/main-compatible.patch
```

若 main 已更新，应先比较差异，不要覆盖新改动。将通过测试的候选分支通过 PR 合到 main。
如果从明坤上传的ZIP对应版本开始，也提供 `mingkun-compatible.patch`：

```bash
git switch -c integration/mingkun-compatible
git apply --check /path/to/mingkun-compatible.patch
git apply /path/to/mingkun-compatible.patch
```

两种补丁只选一种：main补丁用于main基线，mingkun补丁用于明坤基线。
后一种会把main已修正的数据、正式模型与报告一起同步过来，因此补丁更大。
若明坤分支已发生后续变更，先同步并重新核对；不要强行应用补丁或全部选一边。
本次没有连接或修改远程仓库，也没有执行实际 PR 合并。

合并后只需在最新 main 快速再跑验证与核心网页流程，不要求重新训练或重写长报告。
