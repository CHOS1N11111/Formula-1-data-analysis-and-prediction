# Formula 1 数据分析与预测

本项目构建了一个可复现的 Formula 1 数据工程、数据分析、可视化与预测流程。项目从历史数据集和 Jolpica-F1 API 数据出发，构建可用于建模的比赛表，生成统计分析结果和图表，训练多种机器学习与深度学习模型，并模拟 2026 赛季车手和车队总冠军。

项目关注完整的数据流程，而不是单一模型：数据获取、数据验证、数据库检查、特征工程、统计分析、可视化、模型评估、概率校准、积分规则映射、Monte Carlo 赛季模拟，以及最终 2026 冠军预测。

## 项目目标

主要目标包括：

- 构建从原始数据到分析表和建模表的可复现 Formula 1 数据流程。
- 分析 2019-2025 现代 Formula 1 的比赛规律，同时使用更早历史数据作为背景参考。
- 可视化车手、车队、赛道、可靠性、竞争格局和积分榜变化趋势。
- 训练并比较领奖台预测、Top 10 积分区预测和单站积分预测模型。
- 使用当前 Grand Prix 积分表，将模型输出转换为更真实的 Formula 1 单站积分。
- 在最终预测 2026 冠军前，使用回测、概率校准和不确定性模拟验证方法。
- 仅基于已完成比赛积分榜和赛前特征，预测 2026 车手和车队总冠军概率。

静态历史图表有意排除 2026 数据。2026 数据用于当前赛季状态分析和最终冠军预测。

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
|-- tune_f1_feedback_weight.py
|-- tune_f1_current_form_boost.py
|-- predict_f1_2026_championship.py
|-- requirements.txt
|-- data/
|   |-- processed/
|   |-- analysis/
|   |-- modeling/
|   `-- DATA_DESCRIPTION.md
`-- outputs/
    |-- figures/
    |-- videos/
    `-- OUTPUTS_DESCRIPTION.md
```

以下本地目录包含原始下载数据：

```text
data/raw/
formula-1-race-data-sqlite/
formula-1-fantasy-2021/
formula1-data-1950-2022/
```

这些目录不需要放入 Git 仓库，因为可以通过运行数据流程重新下载或生成。

## 数据来源

项目使用四类主要数据来源：

- Kaggle Formula 1 数据集，用于历史比赛记录和 SQLite 数据检查。
- `Formula1.sqlite`，用于更早历史背景分析。
- Jolpica-F1 Ergast 兼容 API，用于现代 Formula 1 比赛、排位、积分榜、车手和车队数据。
- 项目生成的处理后数据表，用于 2019-2026 现代数据和 2003-2026 扩展建模数据。

处理后数据表和建模表保证数据流程可复现。原始比赛 `points` 会保留用于描述性分析，而建模阶段的积分会根据完赛名次统一换算为当前 Grand Prix 积分规则。

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

在项目根目录运行：

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
python train_f1_finish_bucket_model.py
python visualize_f1_model_results.py
python simulate_f1_season_uncertainty.py
python tune_f1_feedback_weight.py
python tune_f1_current_form_boost.py
python predict_f1_2026_championship.py
python analyze_f1_circuit_archetypes.py
python select_f1_prediction_scenarios.py
```

如果本地已经存在 Kaggle 原始数据集或 Jolpica-F1 原始文件，可以跳过下载步骤。后续脚本会复用已有文件。

## 流程阶段

### 数据获取与验证

- `download_formula1_dataset.py` 下载外部 Formula 1 数据集。
- `inspect_formula1_sqlite.py` 检查 SQLite 数据集的表结构和可用历史表。
- `download_jolpica_f1_data.py` 下载 Jolpica-F1 JSON 数据，包括比赛、结果、排位、积分榜、车手和车队。
- `validate_jolpica_f1_data.py` 检查下载的 Jolpica-F1 文件是否与 API 声明总数一致。

### 数据集构建与特征工程

- `build_f1_model_dataset.py` 将 Jolpica-F1 原始数据转换为现代比赛级建模表。
- `build_f1_features.py` 为 2019-2026 现代数据构建赛前和排位后特征。
- `build_f1_extended_features.py` 将 2003-2017 SQLite 历史记录与 2019-2026 Jolpica-F1 现代数据合并，以增加建模样本量。

### 统计分析

现代分析主要覆盖 2019-2025 完整赛季：

- 数据集概况
- 发车位和排位影响
- 车手和车队表现
- 赛道特征
- 特征相关性分析
- 车队竞争集中度和 HHI
- 赛前强度分箱
- 名次提升分析
- 可靠性和完赛状态分析
- 队友对比
- 赛道波动指数

SQLite 历史分析提供背景参考：

- 历史数据集概况
- 每年比赛数量
- 历史车手胜场
- 历史车队胜场
- 历史发车位影响

### 可视化

可视化脚本生成静态图表和动态视频：

- 2019-2025 现代统计图表
- 历史背景图表
- 模型评估图表
- 特征重要性图表
- 积分规则映射和概率校准图表
- 2026 冠军预测图表
- 车手和车队积分变化动态视频

每张图和每个视频的详细说明记录在 `outputs/OUTPUTS_DESCRIPTION.md` 中。

## 建模任务

项目包含三个主要预测任务。

### 领奖台预测

领奖台任务预测某位车手是否进入前三。该任务比较传统机器学习、高级 boosting、stacking 和深度学习模型：

- Logistic Regression
- Random Forest
- Extra Trees
- Histogram Gradient Boosting
- CatBoost
- LightGBM
- XGBoost
- Stacking Ensemble
- Upgraded MLP
- TabNet

评估方式包括时间顺序切分、2025 回测、逐站 Top 3 候选指标和特征重要性分析。

### Top 10 预测

Top 10 任务预测某位车手是否进入积分区。该任务作为单站排序和赛季模拟的概率信号。

项目还包含 Top 10 概率校准，用于检查预测概率是否接近真实事件概率。这一步在 Monte Carlo 赛季模拟前很重要。

### 单站积分预测

积分任务预测车手的期望单站积分。连续型预测积分可以作为模型信号，但最终单站和赛季输出使用 Formula 1 规则映射积分。

对每一场预测比赛，模型先对车手排序，再映射为：

```text
25, 18, 15, 12, 10, 8, 6, 4, 2, 1
```

每场模拟 Grand Prix 只有十位车手得分。

## Formula 1 积分规则与建模范围

建模阶段的积分目标根据完赛名次统一换算为当前 Grand Prix 积分表：

```text
第 1 名：25
第 2 名：18
第 3 名：15
第 4 名：12
第 5 名：10
第 6 名：8
第 7 名：6
第 8 名：4
第 9 名：2
第 10 名：1
```

建模假设：

- 旧历史积分制度不作为回归目标，避免混用早期 Formula 1 赛季中不兼容的积分尺度。
- Sprint 冲刺赛积分不在本项目范围内。
- 不考虑最快圈附加分。
- 每场比赛视为普通全积分 Grand Prix，默认有十个得分名额。
- 最终积分榜和冠军模拟使用 `rule_mapped_points` 作为主要积分结果。
- 连续型 `predicted_points` 仅作为期望积分辅助信号保留。最终 2026 单站排序使用赛内原始预测积分百分位加校准 Top 10 概率，从而减少截断积分饱和，同时保持官方规则映射积分不变。
- 原始历史 `points` 仍用于描述性分析和可视化。

## 最终 2026 冠军预测

最终 2026 预测从已完成的 2026 比赛结果和当前积分榜出发。剩余比赛只使用赛前模型进行预测。

在当前项目输出中，2026 数据状态已经更新到加拿大大奖赛。最终预测因此把 Monaco 之前的比赛作为已完成的当前赛季背景，并从 Monaco Grand Prix 开始预测剩余比赛。

最终预测比较三个排名靠前的赛前模型场景：

- 场景 1：`xgboost_classifier + mlp_regressor`
- 场景 2：`lightgbm_classifier + ridge_regression`
- 场景 3：`hist_gradient_boosting + catboost_regressor`

每个场景都会输出：

- 车手总冠军概率
- 车队总冠军概率
- 车手预测积分不确定性区间
- 车队预测积分不确定性区间
- 剩余比赛的确定性规则映射积分
- 基于冠亚军排序分差的单站确定性冠军置信度
- 场景级冠军对比和诊断结果

主输出使用场景 1，同时 by-model 输出文件和图表保留三个场景，便于比较模型选择的稳健性。

### 最终预测输入

最终冠军预测采用 `pre_race` 特征模式，因为未来比赛发生前无法获得真实排位名次和发车位。

模型输入包括：

- 车手赛前积分和排名
- 车队赛前积分和排名
- 车手近期平均积分
- 车手近期平均完赛名次
- 车手近期领奖台次数
- 车队近期平均积分
- 车队近期领奖台次数
- 车手和车队历史记录数量
- 赛道历史比赛数量
- 赛道杆位胜率
- 赛道前三发车领奖台率
- 赛道平均位置变化
- 赛道大幅位置提升率
- 赛道非前排发车夺冠率
- `driver_id`
- `constructor_id`
- `circuit_id`

这种设计符合真实赛前冠军预测场景，因为它避免使用未来不可获得的排位或发车位信息。

剩余比赛预测支持阻尼式未来特征反馈。在将反馈权重回测与当前赛季在线训练策略对齐后，最终选择的反馈权重为 `0.00`，由 `tune_f1_feedback_weight.py` 在应用 Top 10 概率校准后，使用 2022-2025 历史回测和车手/车队最终积分综合 MAE 选出。这意味着正式预测不会把预测出的未来比赛结果写回后续赛前特征；历史回测显示，在已知当前赛季样本已经加入训练后，零反馈比正反馈更稳定。

最终模型还使用保守的当前赛季在线训练。已完成的 2026 比赛样本会在最终模型训练集中重复加入一次。该设置由 `tune_f1_current_form_boost.py` 通过 2022-2025 回测选出：`online1_boost0.00` 将短历史车手 MAE 从 `33.159524` 降低到 `24.448214`，同时让整体综合 MAE 基本接近 baseline。显式 `ranking_score` form boost 没有放入正式预测，因为它在短历史强状态样本上表现不稳定。

主要局限：

- 数据集中没有天气、练习赛速度、赛车升级、罚退、伤病、车手替换、轮胎分配或安全车概率。
- 车手和车队标识能帮助模型学习历史强弱关系，但如果新赛季竞争格局突然变化，模型反应可能偏慢。
- 赛道历史特征提供了排位重要性和超车波动等技术背景，但不能完整表示赛道布局、轮胎退化、温度或车队赛车与赛道的匹配程度。

## 主要输出

处理后的数据集：

```text
data/processed/f1_model_dataset.csv
data/processed/f1_features.csv
data/processed/f1_model_dataset_extended.csv
data/processed/f1_features_extended.csv
data/processed/f1_2026_schedule.csv
```

分析输出：

```text
data/analysis/
```

建模输出：

```text
data/modeling/
```

重要最终预测输出：

```text
data/modeling/season_prediction_driver_standings_2026.csv
data/modeling/season_prediction_constructor_standings_2026.csv
data/modeling/season_prediction_race_points_2026.csv
data/modeling/season_prediction_model_scenarios_2026.csv
data/modeling/season_prediction_race_signal_diagnostics_2026.csv
data/modeling/season_prediction_model_scenario_diagnostics_2026.csv
data/modeling/season_prediction_summary_2026.json
data/modeling/current_form_boost_backtest_summary.csv
data/modeling/current_form_boost_backtest_summary.json
```

图表：

```text
outputs/figures/
outputs/figures/figure_manifest.csv
outputs/figures/figure_manifest.json
outputs/figures/model_figure_manifest.json
```

视频：

```text
outputs/videos/
outputs/videos/video_manifest.json
```

更详细的文件级说明见：

```text
data/DATA_DESCRIPTION.md
outputs/OUTPUTS_DESCRIPTION.md
```

## 备注

- 静态报告式历史图表使用 2019-2025 现代历史数据和 SQLite 历史数据。
- 静态历史图表不包含 2026 数据。
- 2026 数据用于当前赛季状态分析和最终冠军预测。
- 原始下载数据被 Git 忽略。
- 生成的视频文件保留在仓库中。
