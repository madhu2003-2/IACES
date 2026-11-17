# Resume-ready project summary

Use wording that matches the work you can demonstrate and the data provenance you can verify.

## Project entry

**FitPredict AI — Workout Estimation and Progress Tracking System** | Python, Scikit-learn, Flask, Streamlit, SQLite, MySQL, FoodData Central, uv

- Built an end-to-end machine-learning application with a reusable Scikit-learn preprocessing pipeline, Flask prediction API, and Streamlit dashboard.
- Added schema normalization and configurable single- or multi-dataset training for three workout datasets.
- Implemented leave-one-dataset-out evaluation with per-source MAE, RMSE, and R² reports for multi-source runs.
- Compared Linear Regression, Decision Tree, Gradient Boosting, and tuned Random Forest regressors; recorded source hashes, row counts, and validation protocol in a training manifest.
- Added input validation, local SQLite or MySQL workout history, API documentation, Postman requests, and a model-information endpoint with evaluation metrics and global feature importance.
- Built a goal-profile and longitudinal check-in workflow for dated body weight, workout duration/calories, calorie and protein intake, and push-up/sit-up counts, with SQLite persistence, progress charts, and CSV export.
- Integrated a local FoodData Central survey catalog for food search, serving/gram-based calorie and protein estimates, and dated nutrition logging with daily totals.
- Trained a separate FNDDS text-and-category regression experiment for kcal and protein per 100 g, using description-grouped holdout evaluation and comparison against a category-median baseline.
- Added schema-aware default dataset discovery so unrelated weight-tracking or strength datasets are not accidentally passed into the calorie-regression training workflow.
- Kept personalized calorie/protein targets and goal-duration predictions disabled; the food catalog supports lookup estimates but does not provide longitudinal outcomes or individualized nutrition needs.

## Optional measured-result bullet

Reduced pooled workout-calorie MAE from 236.99 to 83.07 kcal (64.9%) across 19,837 sessions by comparing baselines and regressors inside dataset-held-out validation. Reported each source separately, including the 408.91 kcal MAE on gym sessions. The final duration-only model family scored 89.56 kcal pooled MAE.

## Interview talking points

- Explain why the preprocessing and estimator are saved together as one pipeline.
- Describe which columns were excluded and why `User_ID` should not be a predictive feature.
- Explain how leave-one-dataset-out evaluation works and why source-specific results can differ when the calorie labels or populations are not comparable.
- Explain why the latest source-held-out results show that these datasets do not currently support a reliable shared calorie model.
- Explain how the tracker can collect longitudinal records and what outcome labels and evaluation would be needed before training a goal-time model.
- Describe the next steps: verify each source's target definition and provenance, then decide whether a shared model or separate per-source models are appropriate.
- Show the API request flow and how the optional database failure is kept separate from prediction success.
- Explain why the food composition data supports a transparent lookup calculation, not supervised training for workout calories or personalized goal timelines.
