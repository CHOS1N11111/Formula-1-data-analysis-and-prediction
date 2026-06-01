# Formula 1 数据分析与预测项目

本项目是《大数据与数据工程》课程大作业，围绕 Formula 1 历史赛事数据完成数据获取、数据校验、数据清洗、特征工程、统计分析、可视化展示、视频生成和机器学习预测建模。

项目当前重点包括：

- 2019-2026 现代 F1 赛事数据获取与处理
- 1950-2018 历史 F1 数据背景分析
- 2019-2025 现代 F1 数据统计分析与可视化
- 车手、车队、赛道、排位、可靠性等多角度分析
- 领奖台、Top 10 积分区、单站积分预测模型
- 预测结果可视化与模型诊断

报告用静态图表不包含 2026 数据；2026 数据用于当前赛季状态分析和后续预测任务。

## 项目结构

```text
.
|-- download_formula1_dataset.py
|-- inspect_formula1_sqlite.py
|-- download_jolpica_f1_data.py
|-- validate_jolpica_f1_data.py
|-- build_f1_model_dataset.py
|-- build_f1_features.py
|-- build_f1_extended_features.py
|-- analyze_f1_basic_stats.py
|-- analyze_f1_historical_sqlite.py
|-- visualize_f1_analysis.py
|-- animate_f1_points.py
|-- train_f1_podium_model.py
|-- train_f1_podium_deep_model.py
|-- score_f1_podium_models.py
|-- train_f1_points_model.py
|-- visualize_f1_model_results.py
|-- README.md
|-- README_zh.md
|-- requirements.txt
|-- data/
|   |-- processed/
|   |-- modeling/
|   |-- analysis/
|   `-- DATA_DESCRIPTION.md
`-- outputs/
    |-- figures/
    |-- videos/
    `-- OUTPUTS_DESCRIPTION.md
```

## 数据来源

项目使用的数据来源包括：

- Kaggle Formula 1 相关数据集
- 本地 `Formula1.sqlite` 历史赛事数据库
- Jolpica-F1 提供的 Ergast 兼容 API
- 项目脚本生成的 2019-2026 现代赛事宽表和特征表

其中：

- SQLite 历史数据用于历史背景分析；
- Jolpica-F1 数据用于补充 2019-2026 现代赛事数据；
- 2019-2025 数据用于报告中的现代 F1 可视化分析；
- 2026 数据用于当前赛季状态和后续预测准备。

## 环境依赖

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

生成 MP4 视频需要安装 `ffmpeg`，并确保命令行可以访问：

```powershell
ffmpeg -version
```

## 推荐运行顺序

在项目根目录依次运行：

```powershell
python download_formula1_dataset.py
python inspect_formula1_sqlite.py
python download_jolpica_f1_data.py
python validate_jolpica_f1_data.py
python build_f1_model_dataset.py
python build_f1_features.py
python build_f1_extended_features.py
python analyze_f1_basic_stats.py
python analyze_f1_historical_sqlite.py
python visualize_f1_analysis.py
python animate_f1_points.py
python train_f1_podium_model.py
python train_f1_podium_deep_model.py
python score_f1_podium_models.py
python train_f1_points_model.py
python visualize_f1_model_results.py
```

如果本地已经存在 Kaggle 数据集或 Jolpica-F1 原始数据，可以跳过对应下载步骤。

## 主要输出

处理后的数据：

```text
data/processed/f1_model_dataset.csv
data/processed/f1_features.csv
data/processed/f1_model_dataset_extended.csv
data/processed/f1_features_extended.csv
data/processed/f1_2026_schedule.csv
```

统计分析结果：

```text
data/analysis/
```

机器学习输出：

```text
data/modeling/
```

静态图表：

```text
outputs/figures/
outputs/figures/figure_manifest.csv
outputs/figures/figure_manifest.json
outputs/figures/model_figure_manifest.json
```

动态视频：

```text
outputs/videos/
outputs/videos/video_manifest.json
```

输出说明文档：

```text
data/DATA_DESCRIPTION.md
outputs/OUTPUTS_DESCRIPTION.md
```

## 分析模块

现代数据分析包括：

- 数据集整体概况
- 发车位置与正赛结果关系
- 排位名次与正赛结果关系
- 车手表现汇总
- 车队表现汇总
- 赛道特征分析
- 特征相关性分析
- 车队竞争集中度分析
- 赛前强弱分箱分析
- 名次提升能力分析
- 可靠性与完赛状态分析
- 队友内部对比分析
- 赛道波动指数分析
- 2026 当前赛季状态表

历史背景分析包括：

- 历史数据概况
- 1950-2018 每年比赛数量
- 历史胜场最多车手
- 历史胜场最多车队
- 历史发车位置与完赛结果关系

## 机器学习模块

机器学习部分包括：

- 领奖台预测：以 `is_podium` 为分类目标
- Top 10 积分区预测：以 `is_top10` 为分类目标
- 单站积分预测：以 `current_rule_points` 为回归目标
- 2003-2024 训练、2025 测试的时间顺序回测
- 2022-2025 rolling backtest
- 传统 ML、高级 Boosting、Stacking、MLP、TabNet 模型对比
- 领奖台模型综合评分
- Top 10 概率校准
- 单站积分规则映射
- 特征重要性分析
- 逐站预测误差分析
- 高分样本修正实验

## 积分预测规则口径

数据提取、统计分析和可视化保留原始 `points`，用于反映真实历史数据。

预测建模阶段统一使用当前 F1 Grand Prix 正赛积分规则，根据完赛名次重新计算积分：

```text
25, 18, 15, 12, 10, 8, 6, 4, 2, 1
```

规则说明：

- 不直接使用 2003-2009 等历史赛季的旧积分制度作为回归目标；
- 不考虑最快圈附加 1 分；
- 不考虑 Sprint 冲刺赛积分；
- 默认每场比赛有 10 个得分名额；
- 后续赛季积分榜、车手总冠军和车队总冠军预测以 `rule_mapped_points` 作为主结果；
- 连续 `predicted_points` 只作为期望积分参考；
- 赛前积分、近三场平均积分等建模特征也按当前积分规则重新计算。

## Git 忽略说明

以下内容为原始数据或可重新生成的大文件，已加入 `.gitignore`：

```text
data/raw/
formula-1-race-data-sqlite/
formula-1-fantasy-2021/
formula1-data-1950-2022/
__pycache__/
catboost_info/
```

这些文件不会进入 Git 仓库，但可以通过脚本重新下载或生成。

## 备注

- 报告用静态图表不包含 2026 数据；
- 2026 数据用于当前赛季状态和后续预测；
- 视频文件已纳入 Git，便于报告和 PPT 直接引用；
- 图表和视频说明见 `outputs/OUTPUTS_DESCRIPTION.md`；
- 数据文件说明见 `data/DATA_DESCRIPTION.md`。
