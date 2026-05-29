# Formula 1 数据分析与预测项目

本项目是《大数据与数据工程》课程大作业项目，围绕 Formula 1 历史赛事数据完成数据获取、数据校验、数据清洗、特征工程、统计分析、可视化展示和动态视频生成。

当前项目重点是数据工程处理、数据分析和可视化展示。处理后的数据也为后续“领奖台预测”和“赛季排名预测”任务预留了建模基础。

## 项目内容

本项目已经完成：

- Kaggle 数据集下载；
- Jolpica-F1 API 数据下载；
- 原始 JSON 数据完整性校验；
- SQLite 数据库结构检查；
- 建模宽表构建；
- 赛前特征工程；
- 2019-2025 现代 F1 统计分析；
- 2026 当前赛季状态表；
- 1950-2017/2018 历史背景分析；
- 2019-2025 静态图表可视化；
- 1950-2025 / 2019-2025 动态积分视频生成。

注意：报告用静态图表不包含 2026 数据。2026 数据保留用于当前赛季状态分析和后续预测任务。

## 项目结构

```text
.
|-- download_formula1_dataset.py
|-- inspect_formula1_sqlite.py
|-- download_jolpica_f1_data.py
|-- validate_jolpica_f1_data.py
|-- build_f1_model_dataset.py
|-- build_f1_features.py
|-- analyze_f1_basic_stats.py
|-- analyze_f1_historical_sqlite.py
|-- visualize_f1_analysis.py
|-- animate_f1_points.py
|-- train_f1_podium_model.py
|-- project_log.md
|-- README.md
|-- README_zh.md
|-- requirements.txt
|-- data/
|   |-- processed/
|   |-- modeling/
|   `-- analysis/
`-- outputs/
    |-- figures/
    |-- videos/
    `-- OUTPUTS_DESCRIPTION.md
```

## 数据来源

项目使用的数据来源包括：

- Kaggle Formula 1 相关数据集；
- 本地 `Formula1.sqlite` 历史赛事数据库；
- Jolpica-F1 提供的 Ergast 兼容 API；
- 项目脚本生成的 2019-2026 现代赛事宽表和特征表。

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
python analyze_f1_basic_stats.py
python analyze_f1_historical_sqlite.py
python visualize_f1_analysis.py
python animate_f1_points.py
python train_f1_podium_model.py
```

如果本地已经存在 Kaggle 数据集或 Jolpica-F1 原始数据，可以跳过对应下载步骤。

## 主要输出

处理后的数据：

```text
data/processed/f1_model_dataset.csv
data/processed/f1_features.csv
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
```

动态视频：

```text
outputs/videos/
outputs/videos/video_manifest.json
```

输出说明文档：

```text
outputs/OUTPUTS_DESCRIPTION.md
```

## 分析模块说明

现代数据分析包括：

- 数据集整体概况；
- 发车位置与正赛结果关系；
- 排位名次与正赛结果关系；
- 车手表现汇总；
- 车队表现汇总；
- 赛道特征分析；
- 特征相关性分析；
- 车队竞争集中度分析；
- 赛前强弱分箱分析；
- 名次提升能力分析；
- 可靠性与完赛状态分析；
- 队友内部对比分析；
- 赛道波动指数分析；
- 2026 当前赛季状态表。

机器学习部分包括：

- 以“是否登上领奖台”为分类目标；
- 使用 2019-2024 训练、2025 测试的时间顺序回测；
- 对比逻辑回归、随机森林、极端随机树和直方图梯度提升模型；
- 输出混淆矩阵、特征重要性和逐站 Top 3 领奖台候选命中情况。

历史背景分析包括：

- 历史数据概况；
- 1950-2018 每年比赛数量；
- 历史胜场最多车手；
- 历史胜场最多车队；
- 历史发车位置与完赛结果关系。

可视化部分包括：

- 2019-2025 现代历史数据图表；
- 1950-2018 历史背景图表；
- 赛道排位重要性图；
- 车队竞争集中度图；
- 车手和车队表现趋势图；
- 车队和车手积分动态视频。

## Git 忽略说明

以下内容为原始数据或可重新生成的大文件，已加入 `.gitignore`：

```text
data/raw/
formula-1-race-data-sqlite/
formula-1-fantasy-2021/
formula1-data-1950-2022/
__pycache__/
```

这些文件不会进入 Git 仓库，但可以通过脚本重新下载或生成。

## 备注

- 报告用静态图表不包含 2026 数据；
- 2026 数据用于当前赛季状态和后续预测；
- 视频文件已纳入 Git，便于报告和 PPT 直接引用；
- 详细过程记录见 `project_log.md`；
- 图表和视频说明见 `outputs/OUTPUTS_DESCRIPTION.md`。
