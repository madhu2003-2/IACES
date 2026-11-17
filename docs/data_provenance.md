# Dataset provenance and license

## Local files

The project currently has three workout CSV files in `data/processed/`:

| File | Rows | Source status |
|---|---:|---|
| `calories.csv` | 15,000 | Original publisher and license not verified |
| `exercise_dataset.csv` | 3,864 | Original publisher and license not verified |
| `gym_members_exercise_tracking.csv` | 973 | Original publisher and license not verified |

The current normalizer maps each file to gender, age, weight, duration in minutes, heart rate, and a calorie target. It converts `Session_Duration (hours)` to minutes. A local inspection found no rows dropped for missing or invalid required values. Each training run records the exact input file name, SHA-256 hash, source columns, row counts, and dropped-row count in `reports/training_manifest.json`.

Column names and row counts can suggest where a dataset came from, but they do not prove the publisher, collection method, license, or target-label definition. A previous note identified possible matches for the 15,000-row file in the literature and on Kaggle; treat those only as leads until the download history or original publisher confirms the match. The project does not currently have verified attribution or license metadata for any of the three local files.

## Combining sources

Training defaults to all CSV files in `data/processed/`. When more than one file is selected, evaluation leaves one entire dataset out per fold. The excluded file is not used for model fitting or hyperparameter selection in that fold. `reports/model_source_comparison.csv` records metrics for every model and held-out file.

The calorie target distributions differ across the files. Source-held-out metrics therefore show transfer between these local datasets, not proof that the labels share an identical measurement protocol. Review the source and target definitions before interpreting pooled training results or redistributing data.

## Target and limitations

The `Calories`, `Calories Burn`, or `Calories_Burned` column is treated as the regression target and displayed by the app in kcal. The local files do not establish whether the values are measured, estimated, or generated. Do not describe them as clinically measured or report the model as medically accurate without independent evidence.

The comparison uses gender, age, weight, duration and heart rate as available inputs. The selected serving model uses duration alone. It excludes identifiers and other columns such as height, body temperature, workout type, intensity, and BMI. The API accepts `Male` and `Female`, matching the current training data.

## Nutrition reference catalog

The project also reads the locally supplied FoodData Central survey-food CSV export under `data/processed/`. It uses `food.csv` for descriptions, `food_nutrient.csv` for composition values, `nutrient.csv` to identify protein (nutrient number 203) and energy in kcal (nutrient number 208), and `food_portion.csv` for portion gram weights. In this export, `food_nutrient.nutrient_id` corresponds to `nutrient.nutrient_nbr`; it must not be joined to `nutrient.id` for these lookups. Estimates scale the per-100-g values by the logged gram amount.

This catalog is not used to train the workout calorie model. It contains no workout session or longitudinal body-weight outcome labels. It also does not establish individualized calorie/protein targets or a timeline for reaching a user's goal. Confirm the original download source and license before redistributing these files.

The separate FNDDS nutrition model trains on food descriptions and food category IDs to estimate the catalog's energy and protein values per 100 g. The training script uses a grouped 80/20 holdout and records SHA-256 hashes for the food and nutrient CSV inputs. This experiment is separate from the workout calorie model; see [the nutrition model card](nutrition_model_card.md) for measured results and limitations.
