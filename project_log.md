# Formula 1 数据分析与预测项目记录

## 项目方向

本项目拟以 Formula 1 历史赛事数据为基础，完成一个包含数据获取、数据存储、数据清洗、统计分析、预测建模和可视化展示的课程大作业。当前暂定方向为：

**基于历史 Formula 1 数据的车手领奖台概率预测与赛季排名分析**

后续题目可根据模型效果和系统功能继续调整。

## 已下载并保存的本地数据

项目初始阶段通过 `download_formula1_dataset.py` 下载并保存了以下数据集：

- `formula-1-race-data-sqlite/Formula1.sqlite`

  - 主要 SQLite 数据库。
  - 包含 `races`、`results`、`qualifying`、`drivers`、`constructors`、`circuits`、`driver_standings`、`constructor_standings`、`laptimes`、`pitstops` 等表。
  - 经检查，数据库比赛年份范围为 1950-2018，其中逐站结果和排位数据主要覆盖到 2017。
- `formula1-data-1950-2022/`

  - CSV 数据集。
  - 包含 1950-2022 年的比赛、车手年度表现、最快圈等汇总数据。
- `formula-1-fantasy-2021/`

  - 2021 年 Formula 1 Fantasy 相关数据。
  - 可作为补充数据，但不作为主要训练数据源。

## 已完成脚本

### 1. `inspect_formula1_sqlite.py`

该脚本用于检查本地 SQLite 数据库结构，主要功能包括：

- 连接 `formula-1-race-data-sqlite/Formula1.sqlite`；
- 输出所有数据表名称；
- 输出每张表的行数和字段名；
- 抽样查看核心表前若干行数据；
- 初步确认数据库是否适合用于后续数据分析和建模。

检查结果显示，SQLite 数据库中包含 Formula 1 赛事分析所需的核心关系表，可以支持历史数据分析、正赛结果分析、排位赛影响分析和基础预测建模。

### 2. `download_formula1_dataset.py`

该脚本用于从 Kaggle 下载已有的 Formula 1 相关数据集，下载的数据包括：

- Formula 1 SQLite 数据库；
- 2021 Fantasy 数据；
- 1950-2022 历史 CSV 数据。

### 3. `download_jolpica_f1_data.py`

为了补充 2019 年之后的现代 Formula 1 逐站数据，新增了该脚本。

该脚本使用 Jolpica-F1 提供的 Ergast 兼容 API 获取数据。Jolpica-F1 是 Ergast API 停止服务后的替代数据源之一，适合获取 Formula 1 赛程、比赛结果、排位结果、积分榜、车手和车队信息。

脚本默认下载年份范围：

```text
2019-2026
```

脚本默认下载的数据类型：

```text
races
results
qualifying
driverStandings
constructorStandings
drivers
constructors
```

新增数据单独保存到：

```text
data/raw/jolpica/
```

该脚本不会修改原始 SQLite 数据库，便于保留原始数据和新增数据之间的边界。

下载脚本后续进行了分页逻辑修正。Jolpica-F1 API 虽然可以请求较大的 `limit` 参数，但服务端实际单页返回上限为 100 条左右。因此脚本现在会读取 API 实际返回的 `MRData.limit`，并据此推进 `offset`，避免只保存第一页数据。

### 4. `validate_jolpica_f1_data.py`

为检查新增 JSON 数据是否完整，新增了该校验脚本。

该脚本会逐年、逐 endpoint 检查：

- 本地 JSON 文件是否存在；
- API 声明的 `total` 记录数；
- 本地各分页中实际解析出的记录数；
- 实际记录数是否等于声明记录数；
- 每个文件包含的页数。

该脚本用于判断 Jolpica-F1 原始数据是否已经完整下载。

### 5. `build_f1_model_dataset.py`

为进入后续数据分析和建模阶段，新增了该数据处理脚本。

该脚本从 `data/raw/jolpica/` 中读取 2019-2026 年 Jolpica-F1 原始 JSON 数据，并将 `results` 与 `qualifying` 按照赛季、轮次和车手 ID 进行关联，生成“每场比赛每位车手一行”的建模宽表。

该脚本输出文件包括：

```text
data/processed/f1_model_dataset.csv
data/processed/f1_2026_schedule.csv
data/processed/build_summary.json
```

其中：

- `f1_model_dataset.csv` 是后续统计分析和机器学习建模的基础数据集；
- `f1_2026_schedule.csv` 是 2026 赛季完整赛程，并标记每站是否已有正赛结果；
- `build_summary.json` 记录宽表生成时间、字段列表、总行数、各年份行数和已完成比赛轮次。

### 6. `build_f1_features.py`

为补充机器学习建模所需的赛前特征，新增了该特征工程脚本。

该脚本读取：

```text
data/processed/f1_model_dataset.csv
```

输出：

```text
data/processed/f1_features.csv
data/processed/feature_summary.json
```

该脚本按 `season` 和 `round` 排序，在每一站比赛开始前，根据历史已完成比赛滚动计算赛前特征。这样可以避免使用当前比赛结束后才知道的信息，降低数据泄露风险。

当前新增特征包括：

```text
driver_pre_race_points
driver_pre_race_rank
constructor_pre_race_points
constructor_pre_race_rank
driver_last3_avg_points
driver_last3_avg_finish_position
driver_last3_podium_count
constructor_last3_avg_points
constructor_last3_podium_count
driver_history_count
constructor_history_count
driver_has_history
constructor_has_history
missing_qualifying
grid_is_zero
```

其中：

- `driver_pre_race_points` 表示该车手本赛季当前比赛前的累计积分；
- `constructor_pre_race_points` 表示该车队本赛季当前比赛前的累计积分；
- `driver_pre_race_rank` 和 `constructor_pre_race_rank` 根据赛前积分计算；
- `driver_last3_*` 使用该车手当前比赛之前最近 3 场比赛表现计算；
- `constructor_last3_*` 使用该车队当前比赛之前最近 3 场比赛表现计算；
- `missing_qualifying` 标记排位名次缺失；
- `grid_is_zero` 标记特殊发车位或维修区起步等情况。

### 7. `analyze_f1_basic_stats.py`

为完成基础数据分析功能，新增了该统计分析脚本。

该脚本读取：

```text
data/processed/f1_features.csv
data/processed/f1_2026_schedule.csv
```

输出目录：

```text
data/analysis/
```

该脚本不进行机器学习预测，只完成基础统计分析，主要包括：

- 数据集整体概况；
- 按年份统计比赛记录、车手、车队和赛道数量；
- 发车位置、排位名次与正赛结果关系分析；
- 车手表现汇总；
- 车手按年份表现汇总；
- 车队表现汇总；
- 车队按年份表现汇总；
- 赛道特征汇总；
- 2026 当前车手积分榜；
- 2026 当前车队积分榜；
- 2026 已完成比赛结果；
- 2026 剩余赛程。

### 8. `analyze_f1_historical_sqlite.py`

为补充历史背景线，新增了该 SQLite 历史分析脚本。

该脚本读取：

```text
formula-1-race-data-sqlite/Formula1.sqlite
```

输出目录仍为：

```text
data/analysis/
```

为了与现代数据分析结果区分，输出文件名均带有 `_historical`。

该脚本只做少量历史背景统计，不进入机器学习建模。统计内容包括：

- SQLite 历史数据概况；
- 1950-2018 每年比赛数量；
- 1950-2017 历史胜场最多车手；
- 1950-2017 历史胜场最多车队；
- 1950-2017 发车位置与完赛名次关系；
- 按发车位置统计历史胜率、领奖台率和前十率。

### 9. `visualize_f1_analysis.py`

为生成课程报告中可直接使用的图表，新增了该可视化脚本。

该脚本读取：

```text
data/analysis/
```

输出目录：

```text
outputs/figures/
```

该脚本基于已有基础统计结果生成 PNG 图表，并额外生成图表清单：

```text
outputs/figures/figure_manifest.csv
outputs/figures/figure_manifest.json
```

图表覆盖内容包括：

- 2019-2025 现代历史数据规模；
- 发车位置与胜率、领奖台率、前十率关系；
- 排位名次与正赛结果关系；
- 历史时期与现代时期发车优势对比；
- 2019-2025 车手总积分排名；
- 2019-2025 车队总积分排名；
- 主要车手年度积分趋势；
- 主要车队年度积分趋势；
- 1950-2018 历年比赛数量变化；
- 历史胜场最多车手；
- 历史胜场最多车队；
- 2019-2025 赛道杆位夺冠率。
- 车手年度积分热力图；
- 车队年度积分占比变化；
- 车队领奖台和胜场分布；
- 车手领奖台和胜场分布；
- 各赛季完赛状态结构；
- 不同赛道平均名次变化。
- 排位名次与完赛名次二维热力图；
- 车队年度排名 bump chart；
- 赛道排位重要性气泡图；
- 车手表现稳定性散点图；
- 车队排位到正赛效率散点图；
- 正赛名次变化分布图。

### 10. `animate_f1_points.py`

为展示车手和车队积分随比赛推进的动态变化，新增了该视频生成脚本。

该脚本读取：

```text
data/processed/f1_features.csv
```

输出目录：

```text
outputs/videos/
```

输出内容：

```text
constructor_points_race_progress_1950_2025.mp4
constructor_all_time_points_progress_1950_2025.mp4
driver_points_race_progress_2019_2025.mp4
video_manifest.json
```

车队视频范围为 1950-2025，不包含 2026 数据。车队视频结合 SQLite 1950-2017、Jolpica-F1 2018 和已处理的 2019-2025 数据。车手视频仍使用 2019-2025 现代数据。

其中：

- `constructor_points_race_progress_1950_2025.mp4`：每个赛季开始时积分重置，展示车队赛季积分榜随分站推进的变化；
- `constructor_all_time_points_progress_1950_2025.mp4`：积分不按赛季重置，展示车队历史总累计积分随时间变化；
- `driver_points_race_progress_2019_2025.mp4`：展示现代车手赛季积分榜随分站推进的变化。

注意：可视化图表不包含 2026 数据。2026 数据仍保留在分析 CSV 中，后续用于赛季预测，但不进入当前报告图表。

## Jolpica-F1 数据下载结果

已运行 `download_jolpica_f1_data.py`，并使用 `validate_jolpica_f1_data.py` 完成完整性校验。

下载结果摘要如下：

| 年份 | 比赛数 races | 正赛结果 results | 排位结果 qualifying | 车手积分榜 driverStandings | 车队积分榜 constructorStandings |
| ---- | -----------: | ---------------: | ------------------: | -------------------------: | ------------------------------: |
| 2019 |           21 |              420 |                 418 |                         20 |                              10 |
| 2020 |           17 |              340 |                 340 |                         23 |                              10 |
| 2021 |           22 |              440 |                 439 |                         21 |                              10 |
| 2022 |           22 |              440 |                 440 |                         22 |                              10 |
| 2023 |           22 |              440 |                 440 |                         22 |                              10 |
| 2024 |           24 |              479 |                 479 |                         24 |                              10 |
| 2025 |           24 |              479 |                 479 |                         21 |                              10 |
| 2026 |           22 |               88 |                 107 |                         22 |                              11 |

其中：

- 2019-2025 年数据可用于补充现代 Formula 1 完整逐站历史数据；
- 2026 年 `races` 数据可用于生成当前赛季赛程；
- 2026 年 `results`、`qualifying`、`driverStandings`、`constructorStandings` 可用于作为本赛季预测的当前状态输入；
- 2026 年正赛结果为 88 条，说明当前数据源中已有部分已完成比赛结果。

下载完成后生成了摘要文件：

```text
data/raw/jolpica/download_summary.json
```

完整性校验结果：

```text
Validation passed: all downloaded Jolpica-F1 files match declared totals.
```

校验过程中曾发现第一次下载的 `results` 和 `qualifying` 多数只保存了第一页数据。原因是 API 实际返回的单页上限为 100 条，而脚本初版用请求参数 `PAGE_LIMIT=1000` 直接推进分页 offset。修正分页逻辑并重新下载后，所有文件的本地实际记录数均已与 API 声明总数一致。

## 当前数据是否够用

当前数据已经基本满足项目第一阶段需求。

可以支持的内容包括：

- Formula 1 历史赛事数据分析；
- 车手和车队长期表现分析；
- 排位赛名次、发车位置与正赛结果之间的关系分析；
- 2019-2025 现代 F1 格局分析；
- 领奖台预测模型训练；
- 2026 赛季当前状态分析；
- 基于剩余赛程的 2026 赛季排名预测。

需要注意的是，原始 SQLite 数据库主要覆盖到 2018 年左右，因此后续建模时应将 SQLite 历史数据与 Jolpica-F1 新增数据进行统一清洗和字段转换。

## 建模宽表构建结果

已运行：

```powershell
python build_f1_model_dataset.py
```

生成的主要文件为：

```text
data/processed/f1_model_dataset.csv
```

该文件目前只使用 Jolpica-F1 的 2019-2026 数据，优先保证现代 Formula 1 格局下的数据链路完整。SQLite 历史数据后续可作为历史背景分析或扩展训练数据继续合并。

宽表字段包括：

```text
season
round
race_name
race_date
race_time
circuit_id
circuit_name
circuit_locality
circuit_country
driver_id
driver_code
driver_name
driver_nationality
constructor_id
constructor_name
constructor_nationality
grid
qualifying_position
finish_position
points
laps
status
is_podium
is_top10
data_source
```

构建结果摘要：

| 年份 | 行数 |
|---|---:|
| 2019 | 420 |
| 2020 | 340 |
| 2021 | 440 |
| 2022 | 440 |
| 2023 | 440 |
| 2024 | 479 |
| 2025 | 479 |
| 2026 | 88 |

总行数：

```text
3126
```

标签字段统计：

```text
is_podium = 1 的样本数：468
is_top10 = 1 的样本数：1560
```

缺失值初步检查：

```text
qualifying_position 缺失行数：7
```

这 7 行主要来自未正常参加排位、维修区起步、罚退或特殊比赛情况，后续建模时可将其作为缺失排位值处理，或增加单独的缺失标记特征。

2026 赛程文件：

```text
data/processed/f1_2026_schedule.csv
```

该文件包含 2026 赛季 22 站赛程，其中当前已有正赛结果的比赛为 4 站。该文件后续用于区分“已完成比赛”和“待预测比赛”。

## 当前数据盘点与缺失情况

截至当前检查，项目中已有三类数据：

### 1. Kaggle 基础数据

本地已保存：

```text
formula-1-race-data-sqlite/Formula1.sqlite
formula1-data-1950-2022/
formula-1-fantasy-2021/
```

其中 SQLite 数据库覆盖情况为：

```text
races: 1950-2018，共 997 场比赛
results: 1950-2017，共 23777 条正赛结果
qualifying: 1994-2017，共 7516 条排位结果
```

这部分适合用于历史背景分析、数据库表结构展示和长期趋势分析。但因为现代 F1 车队格局变化较大，第一版预测模型优先使用 2019 年之后的数据。

### 2. Jolpica-F1 原始 JSON 数据

本地已保存：

```text
data/raw/jolpica/
```

覆盖年份：

```text
2019-2026
```

覆盖数据类型：

```text
races
results
qualifying
driverStandings
constructorStandings
drivers
constructors
```

完整性校验结果：

```text
Validation passed: all downloaded Jolpica-F1 files match declared totals.
```

说明本地 JSON 文件中实际解析出的记录数已经与 API 声明总数一致。

### 3. 已处理建模数据

本地已生成：

```text
data/processed/f1_model_dataset.csv
data/processed/f1_2026_schedule.csv
data/processed/build_summary.json
```

建模宽表 `f1_model_dataset.csv` 当前共有：

```text
3126 行
```

各年份样本数：

| 年份 | 行数 |
|---|---:|
| 2019 | 420 |
| 2020 | 340 |
| 2021 | 440 |
| 2022 | 440 |
| 2023 | 440 |
| 2024 | 479 |
| 2025 | 479 |
| 2026 | 88 |

当前宽表包含 40 名不同车手、17 支不同车队。

标签字段统计：

```text
is_podium = 1: 468 行
is_top10 = 1: 1560 行
```

字段缺失情况：

```text
qualifying_position 缺失 7 行
其他核心字段无缺失
```

特殊值情况：

```text
grid = 0 的记录有 44 行
```

`grid = 0` 通常表示维修区起步、未知发车位或特殊处罚情况，后续建模时应单独处理。

### 4. 2026 赛季数据状态

`data/processed/f1_2026_schedule.csv` 中包含：

```text
2026 赛季赛程：22 站
已有正赛结果：4 站
剩余待预测或待更新：18 站
```

2026 本地阵容数据包括：

```text
车手：23 名
车队：11 支
```

需要注意：2026 第 5 站加拿大大奖赛当前已有排位数据，但还没有正赛结果。因此在后续预测中，加拿大站应先作为“未完成正赛”的待预测比赛处理，或在重新下载数据后更新为真实结果。

### 5. 当前缺失或待补充内容

当前数据已经足够开始第一版模型训练和回测，但还缺少以下内容：

1. 更丰富的赛前特征  
   当前已经补充了车手赛前积分、车队赛前积分、赛前排名和最近 3 场表现。后续还可以继续扩展最近 5 场表现、赛道历史表现、同一车手在同一赛道历史成绩等特征。

2. 2026 未完成比赛的排位/发车位  
   对于还没有进行排位赛的 2026 剩余比赛，无法直接获得真实排位名次和发车位置。后续预测时需要采用两阶段方案：先基于历史表现估计赛前强弱，或者在每站排位结束后再进行单站预测。

3. 2026 第 5 站及之后的正赛结果  
   当前本地数据只有 2026 前 4 站正赛结果。后续可以重新运行 `download_jolpica_f1_data.py` 和 `build_f1_model_dataset.py` 更新最新比赛结果。

4. 旧 SQLite 与新 API 数据的统一合并  
   当前建模宽表优先使用 2019-2026 数据。若后续希望扩展历史样本，可以再将 SQLite 的 1950-2017 正赛结果转换为同样字段结构后合并。

5. 可视化图表和模型结果  
   目前还没有生成统计图、模型指标、混淆矩阵、特征重要性图和 2026 预测排名表。

## 赛前特征构建结果

已运行：

```powershell
python build_f1_features.py
```

生成的主要文件为：

```text
data/processed/f1_features.csv
```

该文件在 `f1_model_dataset.csv` 的基础上增加了 15 个赛前特征字段，总行数保持不变：

```text
3126 行
```

各年份样本数：

| 年份 | 行数 |
|---|---:|
| 2019 | 420 |
| 2020 | 340 |
| 2021 | 440 |
| 2022 | 440 |
| 2023 | 440 |
| 2024 | 479 |
| 2025 | 479 |
| 2026 | 88 |

特征构建检查结果：

```text
新增特征字段无空值
赛季第 1 站车手赛前积分均为 0
missing_qualifying = 1 的记录数为 7
grid_is_zero = 1 的记录数为 44
```

示例逻辑检查：

```text
2025 第 1 站 Lando Norris 赛前积分为 0
2025 第 2 站 Lando Norris 赛前积分为 25
```

说明脚本正确地使用前一站及更早比赛结果构造赛前累计积分，没有把当前比赛积分提前泄露到特征中。

## 基础统计分析结果

已运行：

```powershell
python analyze_f1_basic_stats.py
```

输出目录：

```text
data/analysis/
```

生成文件包括：

```text
dataset_overview.csv
dataset_overview_by_year.csv
grid_finish_summary.csv
grid_position_summary.csv
qualifying_position_summary.csv
driver_summary.csv
driver_points_by_year.csv
constructor_summary.csv
constructor_points_by_year.csv
circuit_summary.csv
current_2026_driver_standings.csv
current_2026_constructor_standings.csv
completed_2026_results.csv
remaining_2026_schedule.csv
analysis_summary.json
```

数据概况：

```text
记录数：3126
年份范围：2019-2026
比赛数量：156
车手数量：40
车队数量：17
赛道数量：31
领奖台记录数：468
前十记录数：1560
```

发车位置、排位名次与正赛结果关系：

```text
发车位置与完赛名次相关系数：0.6282
排位名次与完赛名次相关系数：0.6422
杆位夺冠率：0.5385
发车前三领奖台率：0.6902
发车前十进入前十率：0.7662
平均完赛名次：10.5221
```

这些结果说明，发车位置和排位名次与正赛完赛名次存在明显正相关，排位表现和发车位置是后续预测模型中的重要基础变量。

车手汇总结果中，2019-2026 数据范围内总积分最高的车手包括：

```text
Max Verstappen
Lewis Hamilton
Charles Leclerc
Lando Norris
Carlos Sainz
```

车队汇总结果中，总积分最高的车队包括：

```text
Mercedes
Red Bull
Ferrari
McLaren
```

已补充年度趋势统计：

```text
data/analysis/driver_points_by_year.csv
data/analysis/constructor_points_by_year.csv
```

其中 `driver_points_by_year.csv` 按赛季和车手统计：

```text
总积分
胜场数
领奖台次数
前十次数
平均完赛名次
当年所属车队
```

`constructor_points_by_year.csv` 按赛季和车队统计：

```text
总积分
胜场数
领奖台次数
前十次数
平均完赛名次
```

这两个文件用于展示 2019-2026 年车手状态变化和车队竞争格局变化，例如 Mercedes、Red Bull、Ferrari、McLaren 的近年积分趋势。

2026 当前状态：

```text
已完成正赛：4 站
剩余待预测或待更新：18 站
```

已生成：

```text
data/analysis/current_2026_driver_standings.csv
data/analysis/current_2026_constructor_standings.csv
data/analysis/remaining_2026_schedule.csv
```

这部分结果后续可作为 2026 赛季预测的当前状态输入。

补充的深度分析输出：

```text
data/analysis/feature_correlation_summary.csv
data/analysis/constructor_competitiveness_by_year.csv
data/analysis/pre_race_strength_bins.csv
data/analysis/circuit_grid_importance_score.csv
data/analysis/driver_position_gain_summary.csv
data/analysis/constructor_position_gain_summary.csv
```

其中：

- `feature_correlation_summary.csv` 统计赛前特征与完赛名次、领奖台、前十结果之间的 Pearson 和 Spearman 相关系数；
- `constructor_competitiveness_by_year.csv` 统计每赛季前 1、前 2、前 3 车队积分占比、HHI 指数和有效车队数量，用于衡量车队竞争集中度；
- `pre_race_strength_bins.csv` 按赛前积分排名和近期状态分箱，统计不同强弱区间的领奖台率、前十率和平均完赛名次；
- `circuit_grid_importance_score.csv` 构造赛道排位重要性综合指标，综合杆位夺冠率、发车前三领奖台率和发车-完赛相关性；
- `driver_position_gain_summary.csv` 统计车手从发车位到完赛名次的平均名次提升、正向提升比例、大幅提升次数和大幅下滑次数；
- `constructor_position_gain_summary.csv` 统计车队层面的平均名次提升能力。

## 历史背景分析结果

已运行：

```powershell
python analyze_f1_historical_sqlite.py
```

输出目录：

```text
data/analysis/
```

生成文件包括：

```text
dataset_overview_historical.csv
races_by_year_historical.csv
driver_wins_historical.csv
constructor_wins_historical.csv
grid_finish_summary_historical.csv
grid_position_summary_historical.csv
analysis_historical_summary.json
```

历史数据范围：

```text
races: 1950-2018，共 997 场比赛
results: 1950-2017，共 23777 条正赛结果
qualifying: 1994-2017，共 7516 条排位结果
drivers: 842 名
constructors: 208 支
circuits: 73 条
```

历史发车位置与正赛结果关系：

```text
发车位置与完赛名次相关系数：0.4274
杆位夺冠率：0.4128
发车前三领奖台率：0.5316
发车前十进入前十率：0.6240
```

历史胜场最多车手包括：

```text
Michael Schumacher
Lewis Hamilton
Alain Prost
Sebastian Vettel
Ayrton Senna
```

历史胜场最多车队包括：

```text
Ferrari
McLaren
Williams
Mercedes
Red Bull
```

该历史背景线与 2019-2026 现代分析线分开处理。历史线用于报告中的案例背景和长期趋势描述，现代线用于后续 2026 当前赛季分析与预测。

## 可视化图表生成结果

已运行：

```powershell
python visualize_f1_analysis.py
```

输出目录：

```text
outputs/figures/
```

共生成 31 张 PNG 图表：

```text
yearly_records_2019_2025.png
grid_position_outcome_rates.png
qualifying_position_outcome_rates.png
grid_advantage_historical_vs_modern.png
top10_driver_points_2019_2025.png
constructor_points_2019_2025.png
driver_points_trend_top6.png
constructor_points_trend_top6.png
historical_races_by_year.png
historical_top10_driver_wins.png
historical_top10_constructor_wins.png
circuit_pole_win_rate_top12.png
driver_points_heatmap_top10.png
constructor_points_share_top6.png
constructor_podiums_wins_2019_2025.png
driver_podiums_wins_2019_2025.png
status_mix_by_year_2019_2025.png
circuit_position_change_2019_2025.png
qualifying_finish_heatmap_2019_2025.png
constructor_rank_bump_chart_2019_2025.png
circuit_grid_importance_bubble_2019_2025.png
driver_consistency_scatter_2019_2025.png
constructor_qualifying_race_efficiency_2019_2025.png
position_change_distribution_2019_2025.png
feature_correlation_heatmap_2019_2025.png
constructor_competitiveness_by_year_2019_2025.png
constructor_hhi_competitiveness_2019_2025.png
pre_race_strength_podium_rate_2019_2025.png
circuit_grid_importance_score_2019_2025.png
top_driver_position_gain_2019_2025.png
top_constructor_position_gain_2019_2025.png
```

同时生成图表说明清单：

```text
outputs/figures/figure_manifest.csv
outputs/figures/figure_manifest.json
```

其中 `figure_manifest` 记录了每张图的文件名、标题、来源 CSV 和用途说明，便于后续写报告时引用。

新增的技术性可视化重点包括：

- `qualifying_finish_heatmap_2019_2025.png`：展示排位名次与完赛名次的二维联合分布；
- `constructor_rank_bump_chart_2019_2025.png`：展示车队年度排名变化轨迹；
- `circuit_grid_importance_bubble_2019_2025.png`：用杆位夺冠率、平均名次变化、比赛场次和发车-完赛相关性综合描述赛道排位重要性；
- `driver_consistency_scatter_2019_2025.png`：从平均完赛名次、完赛名次波动、领奖台率和平均积分分析车手稳定性；
- `constructor_qualifying_race_efficiency_2019_2025.png`：比较车队平均发车位和平均完赛名次，观察排位到正赛转化效率；
- `position_change_distribution_2019_2025.png`：展示正赛中车手相对发车位的整体名次变化分布。
- `feature_correlation_heatmap_2019_2025.png`：展示赛前特征与完赛名次、领奖台、前十结果之间的相关性；
- `constructor_competitiveness_by_year_2019_2025.png`：展示每赛季前 1、前 2、前 3 车队积分占比，用于分析竞争集中度；
- `constructor_hhi_competitiveness_2019_2025.png`：使用 HHI 指数和有效车队数量分析车队竞争集中度；
- `pre_race_strength_podium_rate_2019_2025.png`：展示赛前排名和近期状态分箱下的领奖台率差异；
- `circuit_grid_importance_score_2019_2025.png`：展示赛道排位重要性综合指标排名；
- `top_driver_position_gain_2019_2025.png`：展示平均名次提升最高的车手；
- `top_constructor_position_gain_2019_2025.png`：展示平均名次提升最高的车队。

这些图表可以直接用于报告中的：

- 案例背景介绍；
- 数据集描述；
- 数据分析结果；
- 后续预测模型依据说明。

## 动态视频生成结果

已运行：

```powershell
python animate_f1_points.py
```

输出目录：

```text
outputs/videos/
```

已生成 2 个 MP4 视频：

```text
constructor_points_race_progress_1950_2025.mp4
constructor_all_time_points_progress_1950_2025.mp4
driver_points_race_progress_2019_2025.mp4
```

视频说明：

- `constructor_points_race_progress_1950_2025.mp4`：展示 1950-2025 每站比赛结束后车队赛季累计积分排名变化，积分每赛季重置；
- `constructor_all_time_points_progress_1950_2025.mp4`：展示 1950-2025 每站比赛结束后车队历史总累计积分排名变化，积分不重置；
- `driver_points_race_progress_2019_2025.mp4`：展示 2019-2025 每站比赛结束后车手累计积分排名变化。

车队视频各包含 1149 帧，对应 1950-2025 年可用数据中的 1149 场比赛；车手视频包含 152 帧，对应 2019-2025 年的 152 场比赛。视频不包含 2026 数据。

同时生成：

```text
outputs/videos/video_manifest.json
```

该文件记录了视频文件名、标题、数据范围、帧数和说明。

## 后续预测方案

### 1. 预测目标

本项目后续预测部分不直接一步预测全年总冠军，而是采用“单站预测再汇总”的方式：

```text
单站车手表现预测 -> 单站积分估计 -> 赛季积分累加 -> 赛季总排名预测
```

核心预测目标包括：

- 预测车手在单站比赛中是否登上领奖台；
- 预测车手在单站比赛中是否进入前十；
- 估计车手单站获得的积分；
- 基于已完成比赛真实积分和剩余比赛预测积分，推算 2026 赛季车手总排名。

其中，第一阶段优先完成“是否登上领奖台”的二分类预测任务，因为该任务目标清晰、结果容易解释，也适合作为课程大作业中的机器学习建模模块。

### 2. 数据建模思路

后续会将 SQLite 历史数据和 Jolpica-F1 新增数据整合为统一的“每场比赛每位车手一行”的建模宽表。

基础字段计划包括：

```text
season
round
race_name
race_date
circuit_id
circuit_name
driver_id
driver_name
constructor_id
constructor_name
grid
qualifying_position
finish_position
points
is_podium
is_top10
```

其中：

```text
is_podium = 1 if finish_position <= 3 else 0
is_top10 = 1 if finish_position <= 10 else 0
```

后续会进一步构造赛前特征，避免使用比赛结束后才知道的信息。

### 3. 可用特征设计

第一版模型计划使用以下特征：

- 赛季年份；
- 比赛轮次；
- 赛道；
- 车手；
- 车队；
- 排位赛名次；
- 发车位置；
- 车手赛前积分；
- 车手赛前积分榜排名；
- 车队赛前积分；
- 车队赛前积分榜排名；
- 车手过去若干场平均完赛名次；
- 车手过去若干场平均积分；
- 车手过去若干场领奖台次数；
- 车队过去若干场平均积分；
- 赛道历史杆位转化率；
- 赛道历史领奖台集中度。

需要注意，建模时不能使用如下赛后信息作为输入特征：

- 正赛最终名次；
- 正赛积分；
- 最快圈排名；
- 进站结果；
- 比赛结束后的积分榜。

这些字段只能作为标签或赛后分析指标使用。

### 4. 模型路线

第一阶段模型采用较容易解释的传统机器学习方法：

- Logistic Regression：作为可解释的线性基线模型；
- Decision Tree：用于展示规则型分类逻辑；
- Random Forest：作为主力分类模型；
- Gradient Boosting：作为进一步提升模型效果的候选模型。

主要分类任务：

```text
输入：赛前信息 + 排位/发车信息 + 车手/车队近期状态
输出：该车手是否登上领奖台
```

评价指标计划包括：

- Accuracy；
- Precision；
- Recall；
- F1-score；
- ROC-AUC；
- 混淆矩阵；
- 特征重要性。

由于“领奖台”样本比例较低，模型评价不能只看准确率，需要重点关注 Precision、Recall 和 F1-score。

### 5. 回测方案

为了证明模型有效，后续会进行历史赛季回测。

计划采用时间顺序划分训练集和测试集：

```text
训练集：2019-2024
测试集：2025
```

或：

```text
训练集：2019-2023
验证集：2024
测试集：2025
```

这样比随机划分更符合真实预测场景，因为真实预测只能使用过去数据预测未来比赛。

回测内容包括：

- 预测 2025 每站车手是否登上领奖台；
- 与真实领奖台结果对比；
- 统计整体分类指标；
- 分析预测错误案例；
- 输出模型特征重要性，解释影响领奖台结果的关键因素。

### 6. 2026 赛季预测流程

完成模型回测后，将使用 2026 当前赛季数据进行预测。

具体流程：

1. 使用 2019-2025 完整逐站数据训练最终模型；
2. 读取 2026 赛程、车手阵容、车队阵容；
3. 对 2026 已完成比赛，使用真实结果和真实积分；
4. 对 2026 未完成比赛，使用模型预测每位车手的领奖台概率和积分期望；
5. 将已完成比赛真实积分与剩余比赛预测积分相加；
6. 得到预测的 2026 赛季车手总积分和总排名；
7. 输出车手排名预测表、领奖台概率榜和关键可视化图表。

输出结果计划包括：

- 2026 车手当前积分榜；
- 2026 剩余比赛单站领奖台概率；
- 2026 预测赛季总积分；
- 2026 预测车手总排名；
- 模型认为影响排名的主要因素。

### 7. 可视化计划

预测结果将配合图表展示，计划包括：

- 发车位置与完赛名次关系散点图；
- 排位名次与领奖台概率关系图；
- 不同车队近年积分变化折线图；
- 车手历史领奖台次数柱状图；
- 模型混淆矩阵热力图；
- 模型特征重要性图；
- 2026 预测总积分柱状图；
- 2026 车手排名预测表。

如果时间充足，后续可进一步制作 Streamlit 可视化页面，用于交互式查看车手、车队、赛道和预测结果。

### 8. 方案特点

该方案的技术性主要体现在：

- 多源数据获取与整合；
- SQLite 历史数据和 API 新增数据统一建模；
- 基于赛前信息构造机器学习特征；
- 使用时间顺序回测，模拟真实预测场景；
- 将单站预测结果汇总为赛季排名预测；
- 使用模型特征重要性解释 Formula 1 成绩影响因素。

## 下一步计划

建模宽表已经生成，下一步应进入数据清洗、探索性分析和第一版模型训练。

建议后续按以下顺序推进：

1. 编写数据质量检查脚本，统计缺失值、异常发车位、各年份样本数量和标签比例；
2. 生成基础统计图，包括发车位置与完赛名次关系、排位名次与领奖台概率关系、车队近年积分变化；
3. 根据基础统计结果整理报告中的数据分析部分；
4. 继续扩展赛道历史表现、车手同赛道历史表现等基础分析功能；
5. 在基础分析完成后，再进入第一版“是否领奖台”机器学习模型。

## 当前运行方式

以下命令均在项目根目录运行：

```text
C:\Users\30474\Desktop\数据库\Formula1_project
```

### 1. 下载 Kaggle 数据集

如果本地还没有 `formula-1-race-data-sqlite`、`formula1-data-1950-2022`、`formula-1-fantasy-2021` 这几个文件夹，可以运行：

```powershell
python download_formula1_dataset.py
```

该脚本会下载并保存基础 Formula 1 数据集。

### 2. 检查 SQLite 数据库结构

用于查看 `Formula1.sqlite` 中有哪些表、每张表有多少行、字段是什么：

```powershell
python inspect_formula1_sqlite.py
```

该命令适合在报告中说明数据库表结构和数据规模之前运行。

### 3. 下载 Jolpica-F1 新增数据

用于补充 2019-2026 年的现代 Formula 1 数据：

```powershell
python download_jolpica_f1_data.py
```

下载结果会保存到：

```text
data/raw/jolpica/
```

如果已经下载过，重新运行会覆盖对应 JSON 文件。

### 4. 校验 Jolpica-F1 数据完整性

每次下载或更新 Jolpica-F1 数据后，建议运行：

```powershell
python validate_jolpica_f1_data.py
```

如果数据完整，会输出：

```text
Validation passed: all downloaded Jolpica-F1 files match declared totals.
```

如果出现 `incomplete`，说明某些文件没有完整下载，需要重新运行 `download_jolpica_f1_data.py` 或单独补下载对应年份和 endpoint。

### 5. 推荐当前运行顺序

当前阶段推荐按以下顺序运行：

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
```

如果基础 Kaggle 数据集已经存在，可以跳过第一步。

## 数据源参考

[1] Kaggle 数据集，Formula 1 race data SQLite，https://www.kaggle.com/，访问日期：2026年5月24日

[2] Jolpica-F1，Ergast-compatible Formula 1 API，https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md，访问日期：2026年5月24日

[3] Formula 1 官方网站，2026 Race Calendar，https://www.formula1.com/en/racing/2026，访问日期：2026年5月24日

[4] Formula 1 官方网站，2026 Driver Standings，https://www.formula1.com/en/results/2026/drivers，访问日期：2026年5月24日
