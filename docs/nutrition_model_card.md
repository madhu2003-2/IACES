# FNDDS nutrition model card

## Summary

This is a separate experimental model that estimates food energy (kcal) and protein (g) per 100 g from the food description and FoodData Central category ID. It does not train the workout-calorie model or estimate a person's daily nutrition needs or time to reach a weight goal.

## Data

- Source: locally supplied FoodData Central FNDDS 2021–2023 survey-food CSV export.
- Tables used: `food.csv`, `food_nutrient.csv`, and `nutrient.csv`.
- Source foods: 5,432; foods with both usable targets: 5,431.
- Targets: nutrient number 208 for kcal and 203 for protein, both per 100 g.
- Features: text TF-IDF unigrams and bigrams from the food description, plus one-hot encoded `food_category_id`.
- Split: reproducible 80/20 holdout grouped by normalized exact description; 4,344 training rows and 1,087 holdout rows.
- Target values are joined using `food_nutrient.nutrient_id` to `nutrient.nutrient_nbr`; this FNDDS export does not use `nutrient.csv.id` for those records.

## Holdout results

| Target | Model MAE | Model RMSE | Model R² | Category-median MAE | Category-median RMSE | Category-median R² |
|---|---:|---:|---:|---:|---:|---:|
| Energy (kcal / 100 g) | 41.927 | 63.256 | 0.8009 | 39.642 | 69.984 | 0.7563 |
| Protein (g / 100 g) | 2.208 | 3.514 | 0.7883 | 2.100 | 3.878 | 0.7422 |

The text/category model has lower RMSE and higher R² than the category-median baseline, but slightly higher MAE for both targets. This suggests some improvement on larger errors without improving the typical absolute error. Treat the model as an experiment, not a substitute for a known food's catalog values.

## Intended use and limitations

- Demonstrate a separate text-and-category regression workflow for nutrition data.
- Compare predictions with the dataset's category-median baseline.
- Keep the FoodData Central catalog values authoritative when a matching food record exists.
- Do not use the model for personalized meal plans, clinical advice, daily nutrient targets, or goal timelines.
- Food descriptions and category IDs may not uniquely capture recipes, preparation, brand, or portion details.
- The split tests unseen exact descriptions from the same export; it does not validate performance on another database, population, or food-labeling system.

## Reproduction and artifacts

From `C:\IACES`, run:

```powershell
.\.venv\Scripts\python.exe -m iaces.ml.train_nutrition_model --data-dir ".\data\processed\FoodData_Central_survey_food_csv_2024-10-31\FoodData_Central_survey_food_csv_2024-10-31"
```

The model is saved as `models/fndds_nutrition_estimator.joblib`. Metrics, the source hashes, and holdout predictions are written under `reports/fndds_nutrition/`.
