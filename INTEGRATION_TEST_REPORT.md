# 整合与回归测试记录

日期：2026-11-11。对象：本目录整合候选版。测试平台为 Linux CPU，非用户的 Mac。
结论：已有 main 正式实验的推理及主要网页交互通过回归验证；新增实验的真实结果切换
仍待明坤补充完整实验目录。没有远程合并、重新采集或全量重训。

## 已执行

| 检查 | 结果与证据 |
| --- | --- |
| Python 语法解析 | app、src、Scripts 全部通过 |
| 离线隔离、分组、文本、图与 bundle 单元测试 | `python -m unittest discover -s Scripts -p 'test_*.py'`：38项通过 |
| main 正式内容保留 | 对408个数据、artifact、报告和核心代码文件逐个SHA-256比较，零变化 |
| checkpoint 实际加载与推理 | 214条 fixture 全部覆盖；最大 absolute log 预测差 9.5367431640625e-07，小于1e-4容差 |
| 既有优化案例 | 6个案例的最佳预测、候选数量与 main 保存结果一致 |
| Streamlit AppTest | 页面初始化、样本数、年龄中位数默认值、新 Brief 预测、30天外推提示、设置优化、已有视频预测和图开关通过，无应用异常 |
| 新 Brief 输入 | 验证提交输入不含参考视频 views 标签 |
| 可选 bundle 故障处理 | 损坏 checkpoint 的单元测试通过：主实验仍可发现，同时返回明确诊断；直接加载损坏 bundle 仍被拒绝 |

AppTest检查控件和执行流程，不等同于人工浏览器视觉验收。正式PDF保持main原件，未添加
未经验证的探索性实验数值。Windows/Mac系统级兼容仍需当地环境验证。

## 缺失或未执行

- 两份输入 ZIP 均无明坤 full-app-v3 的模型、输入、指标和 app_manifest 完整产物。
- 因此未执行真实附加实验的推理一致性和版本来回切换测试；不能把脚本单元测试当作该项通过。
- 未全量重训；新隔离流程已接到最新main的十模型训练函数，旧八模型指标不可直接复用。
- 旧实验训练源码哈希可能不符，程序继续拒绝不匹配来源；不得直接篡改哈希。
- ZIP没有Git提交历史；补丁只针对提供的main内容。远程main若有后续提交，需重新核对。

## 用户合并后验收

在Mac的项目环境中执行：

```bash
python -m unittest discover -s Scripts -p 'test_*.py'
python Scripts/verify_integration.py --check-app
python -m streamlit run app.py
```

手动检查图、预测、优化、JSON和PDF下载。补齐附加实验后，执行
`python Scripts/validate_app_experiment.py --input <实验目录> --check-app`。
最终main通过这些关键流程后，只需补记测试日期、commit和结果，不必重训或撰写长报告。
