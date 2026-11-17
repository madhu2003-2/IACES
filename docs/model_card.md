# FitPredict AI model card

## Summary

FitPredict AI reports an experimental estimate of workout calories. The current serving model fits a single nonnegative coefficient to duration, with no intercept. Trained on 19,837 sessions, it returns approximately 8.44 kcal per minute. A 45-minute workout produces 379.88 kcal. This coefficient describes a fit to the local training labels. The Flask API and Streamlit dashboard identify which inputs affect the estimate.

## Intended use

- Demonstrate an end-to-end tabular machine-learning workflow and API integration.
- Explore predictions on inputs similar to the training data.
- Compare performance between the workout CSV sources included with the project.

Do not use the estimate for medical, nutritional, clinical, or safety decisions. It is not a validated measurement of an individual's energy expenditure.

## Data and features

- Local files: `data/processed/calories.csv` (15,000 rows), `exercise_dataset.csv` (3,864 rows), and `gym_members_exercise_tracking.csv` (973 rows).
- Default multi-file training selects CSVs under `data/processed/` whose headers match the workout-calorie schema; unrelated weight, strength, or research files are skipped. The training manifest records actual counts and hashes for each selected file.
- Target: the source calorie column (`Calories`, `Calories Burn`, or `Calories_Burned`), displayed by the app as kcal.
- The request accepts gender, age, weight, duration in minutes and average heart rate. Only duration affects the model selected for this release.
- Excluded columns: identifiers and source-specific inputs such as height, body temperature, workout type, intensity, and BMI.
- Local publisher, license, collection method, and target-label generation remain unverified. See [data provenance](data_provenance.md).

## Training and evaluation

For each outer fold the trainer excludes one entire file. On the other files it compares seven candidates by mean inner-fold MAE. The simplest baselines are the median calories in the training sessions and a fitted constant calories-per-minute rate. A third candidate lets calories per minute vary linearly with average heart rate. Both rate models use nonnegative coefficients and zero intercept. The remaining candidates are full-input linear regression, decision tree, gradient boosting and a tuned Random Forest. The full-input linear candidate applies the API's zero floor in validation.

For this run each inner fold holds out one of the remaining sources. The selected candidate is refitted on the available training data and predicts the untouched outer file. The final fit repeats selection using all three files as validation groups, then refits the winner on all rows. With only two input files a training fold containing one source uses shuffled row cross-validation. The single-file training command retains its original seeded random holdout and Random Forest workflow.

| Held-out file | Rows | Training-only choice | Previous forest MAE | Selection procedure MAE | Selection R² | Fixed duration-only MAE |
|---|---:|---|---:|---:|---:|---:|
| `calories.csv` | 15,000 | Duration and heart rate | 218.99 | 36.26 | 0.576 | 55.05 |
| `exercise_dataset.csv` | 3,864 | Duration and heart rate | 216.33 | 182.74 | −2.877 | 143.11 |
| `gym_members_exercise_tracking.csv` | 973 | Duration only | 596.47 | 408.91 | −1.596 | 408.91 |
| All sessions, weighted by row count | 19,837 | Fold-specific | 236.99 | 83.07 | 0.527 | 89.56 |

All MAEs are in kcal. With equal weight for each file, the selection procedure averaged 209.30 kcal MAE and the previous forest averaged 343.93. The selection procedure's pooled RMSE was 144.34 kcal. Predicting the training median for every session produced a pooled MAE of 279.29 kcal.

**Why the app uses duration alone.** Giving each source equal weight during final selection yielded 202.35 kcal MAE for duration alone, 202.80 with heart rate and 217.54 for full-input linear regression. Weighting by individual workouts instead favors the largest file and gave full-input linear regression a lower pooled MAE of 79.53 kcal. The fixed duration-only family had pooled RMSE 135.11 kcal and R² 0.585. Its result and the selection procedure's score measure different quantities.

**Why errors remain large.** Median session lengths were 16, 40 and 75.6 minutes respectively. The within-file correlation of duration with calories was 0.955, 0.022 and 0.908. The exercise file has very little linear association between duration and its target. The gym file has a much higher median calorie label (893 kcal) than the short-workout file (79 kcal). Two held-out sources still have negative selection R². The benchmark has influenced model development. A new source with documented measurement methods is needed to test transfer independently.

## Model information

`GET /model-info` identifies the final candidate, active inputs and evaluation protocol. For multi-source training, `source_selected` denotes predictions from the training-only choice in each outer fold. The comparisons for individual candidates are in `reports/model_source_comparison.csv`. For models without tree importance, the API reports the change in MAE (kcal) after shuffling each input five times on up to 1,500 sampled training rows. This describes the fitted model and is separate from the held-out performance test. Tree models retain impurity-based importance.

## Limitations

- Dataset source, license, collection method, and target-label generation are not documented in the local files.
- The three sources have different calorie and duration distributions; leave-one-dataset-out results measure transfer among these files, not generalization to an independent population or device.
- The model only accepts the two gender categories supported by the training data.
- Accepted values may still be outside the training distribution.
- No prediction interval or calibrated uncertainty estimate is provided.
- Estimates are for demonstration only and are not medical measurements.

## Goal progress tracker

The Streamlit app can save a user-entered goal profile and dated weight, workout, nutrition, and strength records. FoodData Central is used as a local reference lookup for estimated food calories/protein, not as training data for workout calorie or goal-time models. These records are collected for future analysis and are not inputs or targets for the current calorie-burn model. The app does not currently estimate goal completion time or recommend daily calorie/protein targets; local datasets do not establish those outcomes with sufficient consistency for individual advice.

## Reproduction

From `C:\IACES`, run `uv run python -m iaces.ml.train_model` to reproduce the latest all-source model; compatible workout-calorie CSVs under `data/processed` are selected automatically and unrelated CSVs are skipped. Evaluate saved predictions with `uv run python -m iaces.ml.evaluate_model --predictions .\reports\model_holdout_predictions.csv`.
