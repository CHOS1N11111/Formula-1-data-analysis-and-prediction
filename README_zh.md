# Formula 1 数据分析与预测项目

本项目是《大数据与数据工程》课程大作业，围绕 Formula 1 数据完成数据工程、统计分析、可视化和预测建模。项目从原始数据集和 API 数据出发，构建可复现的数据处理流程，生成处理后的宽表、统计分析结果、静态图表、动态视频、机器学习模型、赛季不确定性回测结果以及 2026 赛季冠军预测结果。

当前实现内容包括数据获取、数据验证、特征工程、统计分析、可视化、领奖台预测、Top 10 积分区预测、单站积分预测、F1 积分规则映射、概率校准、Monte Carlo 赛季不确定性模拟，以及最终 2026 车手/车队总冠军概率预测。

## 项目范围

项目覆盖以下内容：

- Kaggle 数据集下载
- Jolpica-F1 API 数据下载
- 原始数据完整性验证
- SQLite 数据库检查
- 建模宽表构建
- 赛前特征工程
- 2019-2025 现代 F1 统计分析
- 2026 当前赛季状态表
- 1950-2017/2018 历史背景分析
- 2019-2025 现代数据和历史数据静态可视化
- 积分榜动态视频
- 领奖台、Top 10 和单站积分预测模型
- F1 规则映射积分策略对比
- 2025 Monte Carlo 赛季不确定性回测
- 2026 车手和车队总冠军概率预测

报告用历史分析静态图表有意排除 2026 数据。2026 数据用于当前赛季状态分析和最终冠军预测。

## 仓库结构

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
|-- train_f1_points_model.py
|-- score_f1_podium_models.py
|-- visualize_f1_model_results.py
|-- simulate_f1_season_uncertainty.py
|-- predict_f1_2026_championship.py
|-- requirements.txt
|-- data/
|   |-- processed/
|   |-- modeling/
|   |-- analysis/
|   `-- DATA_DESCRIPTION.md
`-- outputs/
    |-- figures/
    `-- videos/
```

以下本地目录包含原始下载数据或可重新生成的大文件，已被 Git 忽略：

```text
data/raw/
formula-1-race-data-sqlite/
formula-1-fantasy-2021/
formula1-data-1950-2022/
```

这些文件可以通过运行项目脚本重新生成。

## 数据来源

项目使用的数据来源包括：

- Kaggle Formula 1 数据集
- Formula1.sqlite 历史比赛数据库
- Jolpica-F1 Ergast 兼容 API
- 项目处理后的 2019-2026 现代 Formula 1 比赛数据

数据来源说明和生成结果记录在课程报告、`data/DATA_DESCRIPTION.md` 和 `outputs/OUTPUTS_DESCRIPTION.md` 中。

## 环境配置

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

生成 MP4 视频需要安装 `ffmpeg`，并确保它可以在 `PATH` 中访问。

检查命令：

```powershell
ffmpeg -version
```

## 推荐运行顺序

在项目根目录运行以下命令：

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
python simulate_f1_season_uncertainty.py
python predict_f1_2026_championship.py
```

如果本地已经存在 Kaggle 数据集或 Jolpica-F1 原始数据，可以跳过对应下载步骤。

## 主要输出

处理后的数据集：

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

模型输出：

```text
data/modeling/
```

图表输出：

```text
outputs/figures/
outputs/figures/figure_manifest.csv
outputs/figures/figure_manifest.json
```

视频输出：

```text
outputs/videos/
outputs/videos/video_manifest.json
```

## 分析模块

现代 F1 分析包括：

- 数据集概况
- 发车位和排位对比赛结果的影响
- 车手表现
- 车队表现
- 赛道特征
- 特征相关性分析
- 车队竞争集中度和 HHI 指标
- 赛前强度分箱
- 名次提升分析
- 可靠性和完赛状态分析
- 队友对比
- 赛道波动指数
- 2026 当前赛季状态表

机器学习部分包括：

- 领奖台分类目标
- 2019-2024 训练、2025 测试的时间顺序回测
- 2022-2025 滚动回测
- 结合 SQLite 2003-2017 和 Jolpica-F1 2019-2026 的扩展训练特征
- 两种特征模式：排位后预测，以及不包含发车位/排位字段的赛前预测
- 只使用历史比赛构建的赛道历史特征
- Logistic Regression、Random Forest、Extra Trees、Histogram Gradient Boosting 对比
- 用于概率校准的 Calibrated Random Forest
- 混淆矩阵、特征重要性和逐站 Top 3 领奖台候选评估
- 升级版 MLP 和 TabNet 深度学习模型对比，并输出训练损失曲线
- ML/DL 综合结果可视化
- CatBoost、LightGBM、XGBoost、Stacking Ensemble 和逐站排序指标
- Top 10 和单站积分预测，作为 2026 冠军模拟的中间任务
- 用于单站 F1 计分的规则映射积分策略对比
- 用于赛季模拟的 leave-one-race-out Top 10 概率校准
- 在最终 2026 冠军预测前进行 2025 Monte Carlo 赛季不确定性回测
- 使用赛前模型完成 2026 车手和车队总冠军概率预测，并比较前三模型场景

积分建模规则范围：

- 单站积分回归目标根据完赛名次统一换算为当前 Grand Prix 正赛积分表：`25, 18, 15, 12, 10, 8, 6, 4, 2, 1`。
- 旧历史积分制度不作为回归目标使用，以避免 2003-2009 等赛季与现代 F1 积分尺度混用。
- 不考虑最快圈附加分。
- 不考虑 Sprint 冲刺赛积分。
- 每场比赛视为普通全积分 Grand Prix，默认有十个得分名额。
- 最终积分榜和冠军模拟输出应以 `rule_mapped_points` 作为主要积分结果。连续型 `predicted_points` 仅作为期望积分辅助信号保留。
- 原始 `points` 仍然用于历史分析和可视化。建模阶段的积分型特征，例如赛前积分和近期平均积分，会根据当前积分表从完赛名次重新计算。
- 最终 2026 预测从已完成比赛积分榜出发，剩余比赛只使用赛前特征，不使用未来排位或发车位数据。

历史分析包括：

- 历史数据集概况
- 每年比赛数量
- 历史车手胜场
- 历史车队胜场
- 历史发车位影响

## 备注

- 静态历史报告图表使用 2019-2025 现代历史数据和 SQLite 历史数据。
- 2026 数据不用于静态历史报告图表。
- 2026 数据用于当前赛季状态分析和最终冠军预测。
- 原始下载数据被 Git 忽略。生成的视频文件会保留在仓库中。
