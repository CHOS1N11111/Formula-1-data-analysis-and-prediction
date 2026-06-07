# Final 2026 Formula 1 Championship Prediction Summary

This document summarizes the final 2026 driver and constructor championship prediction outputs. It is intended for report and presentation use.

## Prediction Objective

The final prediction estimates:

- 2026 driver championship probabilities
- 2026 constructor championship probabilities
- projected final points distributions
- deterministic rule-mapped points for each remaining Grand Prix
- model-scenario sensitivity results

The prediction starts from completed 2026 races and forecasts the remaining 17 races using pre-race features only.

## Data Used

The prediction uses the extended feature dataset:

```text
data/processed/f1_features_extended.csv
```

Completed 2026 race results are used as the current-season starting state. Remaining 2026 races are loaded from:

```text
data/analysis/remaining_2026_schedule.csv
```

Current standings and completed-race state are available in:

```text
data/analysis/current_2026_driver_standings.csv
data/analysis/current_2026_constructor_standings.csv
data/analysis/completed_2026_results.csv
```

## Modeling Scope

The final prediction uses `pre_race` features only, because future qualifying positions and grid positions are not available before a race happens.

Important assumptions:

- Sprint races are not modeled.
- Fastest-lap bonus points are not modeled.
- Each race is treated as a normal full-points Grand Prix.
- Exactly 10 drivers score points in each simulated race.
- Completed 2026 results are converted from finish position to the current Grand Prix points table.
- Historical descriptive analysis can keep raw points, but prediction-stage points use the current F1 scoring rule.

## Points Rule

All final race and season outputs use the current Grand Prix points table:

```text
1st  = 25
2nd  = 18
3rd  = 15
4th  = 12
5th  = 10
6th  = 8
7th  = 6
8th  = 4
9th  = 2
10th = 1
```

Each race therefore assigns 101 total points.

## Final Model Setup

The primary scenario is Scenario 1:

```text
Top 10 model: xgboost_classifier
Points model: mlp_regressor
Feature mode: pre_race
Simulation count: 5000
```

The final race ranking strategy is:

```text
0.7 * race-level raw predicted-points percentile
+ 0.3 * calibrated Top 10 probability
```

This strategy reduces capped-points saturation. The displayed `predicted_points` values are still clipped for readability, but race ranking uses the raw model output converted into a within-race percentile signal.

## Primary Prediction Result

### Driver Championship

Primary Monte Carlo prediction:

```text
Predicted champion: Andrea Kimi Antonelli
Champion probability: 0.534800
Runner-up by probability: Lewis Hamilton
Runner-up probability: 0.203000
Probability margin: 0.331800
```

Top driver probabilities:

| Rank | Driver | Team | Current Points | Mean Projected Points | Champion Probability |
|---:|---|---|---:|---:|---:|
| 1 | Andrea Kimi Antonelli | Mercedes | 118.000000 | 288.130000 | 0.534800 |
| 2 | Lewis Hamilton | Ferrari | 61.000000 | 244.315600 | 0.203000 |
| 3 | George Russell | Mercedes | 67.000000 | 236.453000 | 0.144400 |
| 4 | Charles Leclerc | Ferrari | 58.000000 | 208.272800 | 0.059000 |
| 5 | Max Verstappen | Red Bull | 37.000000 | 184.734200 | 0.030400 |

The deterministic projected champion is Lewis Hamilton, while the Monte Carlo probability leader is Andrea Kimi Antonelli. The final conclusion should use the Monte Carlo probability result, because it summarizes 5000 simulated season outcomes rather than one deterministic race-by-race path.

### Constructor Championship

Primary Monte Carlo prediction:

```text
Predicted champion: Mercedes
Champion probability: 0.704600
Runner-up by probability: Ferrari
Runner-up probability: 0.268200
Probability margin: 0.436400
```

Top constructor probabilities:

| Rank | Constructor | Current Points | Mean Projected Points | Champion Probability |
|---:|---|---:|---:|---:|
| 1 | Mercedes | 185.000000 | 524.583000 | 0.704600 |
| 2 | Ferrari | 119.000000 | 452.588400 | 0.268200 |
| 3 | McLaren | 71.000000 | 330.527200 | 0.019600 |
| 4 | Red Bull | 51.000000 | 300.750600 | 0.007600 |
| 5 | Alpine F1 Team | 34.000000 | 148.138400 | 0.000000 |

## Model Scenario Comparison

The final pipeline preserves the original three pre-race model scenarios and adds two season-backtest-derived diagnostic scenarios:

| Scenario | Top 10 Model | Points Model | Driver Champion | Driver Probability | Constructor Champion | Constructor Probability | Role |
|---:|---|---|---|---:|---|---:|---|
| 1 | xgboost_classifier | mlp_regressor | Andrea Kimi Antonelli | 0.534800 | Mercedes | 0.704600 | primary recommended |
| 2 | lightgbm_classifier | ridge_regression | Andrea Kimi Antonelli | 0.654000 | Mercedes | 0.825400 | sensitivity only |
| 3 | hist_gradient_boosting | catboost_regressor | Andrea Kimi Antonelli | 0.673200 | Mercedes | 0.831800 | sensitivity only |
| 4 | hist_gradient_boosting | mlp_regressor | Andrea Kimi Antonelli | 0.555800 | Mercedes | 0.710600 | secondary sensitivity |
| 5 | lightgbm_classifier | xgboost_regressor | Andrea Kimi Antonelli | 0.659600 | Mercedes | 0.809800 | sensitivity only |

Scenario 1 is used as the primary result because it has the best scenario-selection score and more realistic winner diversity. Scenarios 2 and 3 are retained as sensitivity checks, but they are too concentrated because both predict Andrea Kimi Antonelli as winner in all remaining races. Scenario 4 is added from the season-level backtest best-overall row and behaves similarly to Scenario 1, while Scenario 5 is the season-level best non-concentrated candidate but still becomes concentrated in the 2026 forecast.

## Season-Level Model-Combination Backtest

The project includes an additional diagnostic script:

```text
backtest_f1_model_scenarios.py
```

This script does not change the primary Scenario 1 prediction. It evaluates candidate Top 10 and points-model combinations by rolling out historical seasons from 2022 to 2025 after the first five known races.

The current best-ranked diagnostic row by season-level selection score is used as Scenario 4:

```text
Top 10 model: hist_gradient_boosting
Points model: mlp_regressor
Average combined points MAE: 55.655682
Driver champion hit rate: 0.750000
Constructor champion hit rate: 0.750000
Average max winner share: 0.986842
Recommended role: sensitivity_only_high_concentration
```

The best non-concentrated candidate is used as Scenario 5:

```text
Top 10 model: lightgbm_classifier
Points model: xgboost_regressor
Average combined points MAE: 60.138582
Driver champion hit rate: 0.750000
Constructor champion hit rate: 0.750000
Average max winner share: 0.731424
Recommended role: candidate
```

This result is useful because it confirms that strong season-level error metrics can still coexist with excessive deterministic winner concentration. Therefore, the backtest supports keeping concentrated scenarios as sensitivity-only rather than automatically promoting them to the final forecast. Scenario 5 was less concentrated in historical backtests, but under the 2026 current-season state it still predicts Andrea Kimi Antonelli as winner in all remaining races.

## Race-Level Winner Confidence

Each deterministic future-race winner is assigned a confidence label based on the ranking-score gap between the predicted winner and runner-up:

```text
strong: gap >= 0.08
medium: 0.02 <= gap < 0.08
weak: gap < 0.02
```

For the primary Scenario 1:

```text
medium: 16 races
weak: 1 race
strong: 0 races
```

This means most single-race winner predictions have moderate separation, while one race remains low-confidence. Race-level winner confidence explains deterministic single-race picks; it does not alter the Monte Carlo championship probabilities.

## Key Output Files

Final prediction tables:

```text
data/modeling/season_prediction_driver_standings_2026.csv
data/modeling/season_prediction_constructor_standings_2026.csv
data/modeling/season_prediction_race_points_2026.csv
data/modeling/season_prediction_summary_2026.csv
data/modeling/season_prediction_summary_2026.json
```

Scenario comparison outputs:

```text
data/modeling/season_prediction_model_scenarios_2026.csv
data/modeling/season_prediction_model_scenario_diagnostics_2026.csv
data/modeling/season_prediction_scenario_selection_2026.csv
data/modeling/season_prediction_scenario_selection_summary_2026.json
data/modeling/season_model_scenario_backtest_metrics.csv
data/modeling/season_model_scenario_backtest_summary.csv
data/modeling/season_model_scenario_backtest_summary.json
```

By-model outputs:

```text
data/modeling/season_prediction_driver_standings_2026_by_model.csv
data/modeling/season_prediction_constructor_standings_2026_by_model.csv
data/modeling/season_prediction_race_points_2026_by_model.csv
```

Race diagnostics:

```text
data/modeling/season_prediction_race_signal_diagnostics_2026.csv
data/modeling/season_prediction_circuit_archetype_diagnostics_2026.csv
data/modeling/season_prediction_top10_calibration_2026.csv
```

Main figures:

```text
outputs/figures/season_prediction_driver_champion_2026.png
outputs/figures/season_prediction_constructor_champion_2026.png
outputs/figures/season_prediction_driver_points_uncertainty_2026.png
outputs/figures/season_prediction_constructor_points_uncertainty_2026.png
outputs/figures/season_prediction_model_scenarios_2026.png
outputs/figures/season_model_scenario_backtest.png
```

Scenario-specific figures:

```text
outputs/figures/season_prediction_s1_driver_champion_2026.png
outputs/figures/season_prediction_s1_constructor_champion_2026.png
outputs/figures/season_prediction_s1_driver_points_uncertainty_2026.png
outputs/figures/season_prediction_s1_constructor_points_uncertainty_2026.png
outputs/figures/season_prediction_s2_driver_champion_2026.png
outputs/figures/season_prediction_s2_constructor_champion_2026.png
outputs/figures/season_prediction_s2_driver_points_uncertainty_2026.png
outputs/figures/season_prediction_s2_constructor_points_uncertainty_2026.png
outputs/figures/season_prediction_s3_driver_champion_2026.png
outputs/figures/season_prediction_s3_constructor_champion_2026.png
outputs/figures/season_prediction_s3_driver_points_uncertainty_2026.png
outputs/figures/season_prediction_s3_constructor_points_uncertainty_2026.png
outputs/figures/season_prediction_s4_driver_champion_2026.png
outputs/figures/season_prediction_s4_constructor_champion_2026.png
outputs/figures/season_prediction_s4_driver_points_uncertainty_2026.png
outputs/figures/season_prediction_s4_constructor_points_uncertainty_2026.png
outputs/figures/season_prediction_s5_driver_champion_2026.png
outputs/figures/season_prediction_s5_constructor_champion_2026.png
outputs/figures/season_prediction_s5_driver_points_uncertainty_2026.png
outputs/figures/season_prediction_s5_constructor_points_uncertainty_2026.png
```

## How To Interpret The Figures

- `season_prediction_driver_champion_2026.png`: compares driver championship probabilities from the primary scenario.
- `season_prediction_constructor_champion_2026.png`: compares constructor championship probabilities from the primary scenario.
- `season_prediction_driver_points_uncertainty_2026.png`: shows projected driver points intervals. The red marker is current points.
- `season_prediction_constructor_points_uncertainty_2026.png`: shows projected constructor points intervals. The red marker is current points.
- `season_prediction_model_scenarios_2026.png`: compares champion probabilities across all five model scenarios.
- `season_model_scenario_backtest.png`: compares season-level backtest MAE and selection score for candidate model combinations.

The uncertainty interval charts should be interpreted as simulated final-points ranges, not exact forecasts.

## Main Limitations

The prediction is technically rule-consistent, but it still has realistic limitations:

- Future qualifying and grid positions are unavailable, so the model uses pre-race features only.
- Weather, tire strategy, practice pace, penalties, mechanical upgrades, driver injuries, and safety-car risk are not included.
- Short-history drivers are harder to model because the dataset contains fewer past races for them.
- Race-level deterministic winners are point estimates and should not be over-interpreted.
- Scenarios 2 and 3 are useful sensitivity checks but are too concentrated for the main conclusion.

## Final Conclusion

Using the primary pre-race model scenario and Monte Carlo season simulation, the project predicts:

```text
2026 Driver Champion: Andrea Kimi Antonelli
Champion Probability: 0.534800

2026 Constructor Champion: Mercedes
Champion Probability: 0.704600
```

The result should be presented as a probability-based forecast, not a certain outcome.
