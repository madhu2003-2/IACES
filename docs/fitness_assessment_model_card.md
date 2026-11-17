# Experimental fitness assessment model

The model predicts the A–D label in the local `data/processed/bodyPerformance.csv` dataset. It classifies existing physical test results. It does not forecast improvement, prescribe exercises, estimate calories, or determine a weight-goal completion date. The original source, license, grading criteria, and test protocols need independent confirmation before interpreting the label as a personal fitness rating.

## Inputs

Age (21–64), gender (male or female as encoded in the source), height (cm), weight (kg), body fat (%), grip force (kg), sit-and-bend measurement (cm), sit-up count, and standing broad jump (cm). Inputs are constrained to selected training ranges. Measurements should come from the same test protocols as the source; those protocols have not been verified. The model cannot assess people outside the dataset's age or gender categories.

## Training and evaluation

The training script removes invalid values and duplicate selected-feature rows, then uses an 80/20 stratified random row split (seed 42). It compares Random Forest, Extra Trees, and histogram gradient boosting using three-fold cross-validation on training rows only, selecting the highest macro F1 before evaluating the reserved rows. It also reports a majority-class baseline. This holdout was used in the previous version, so a fresh, independent dataset is needed for a stronger evaluation. Results, per-class scores, a confusion matrix, and a data hash are saved in `reports/fitness_assessment_metrics.json`. Accuracy describes this holdout and is not a probability that a particular person's result is correct. A random row split does not establish performance on new populations or ensure person-independent evaluation.

The research weight and screening files and the Gym Progress file lack this A–D target and required physical-test inputs. They were reviewed separately and are not included in this model's training data. See `docs/weight_dataset_audit.md`.

## Reproduction

From `C:\IACES`:

```powershell
.\.venv\Scripts\python.exe -m iaces.ml.train_fitness_assessment
```

This replaces `models/fitness_assessment.joblib` and `reports/fitness_assessment_metrics.json`. Restart Flask after retraining because the API caches the model. The existing calorie model is separate.
