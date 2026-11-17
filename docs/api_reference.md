# API reference

Base URL for local development: `http://127.0.0.1:5000`.

## `POST /fitness-assessment`

Estimates the body-performance dataset's A–D category from existing measurements. The result is experimental and is not medical advice, an exercise prescription, a goal-time estimate, or a forecast of improvement. Source protocols and label definitions need confirmation. JSON example:

```json
{
  "age": 30, "gender": "Male", "height_cm": 170, "weight_kg": 70,
  "body_fat_percent": 23, "grip_force_kg": 35, "flexibility_cm": 15,
  "situps": 35, "broad_jump_cm": 180
}
```

The response contains `category`, `holdout_accuracy`, and `disclaimer`. Invalid inputs return 400; a missing model returns 503. Accuracy describes the dataset holdout, not certainty for an individual. See `docs/fitness_assessment_model_card.md`.

## `GET /health`

Returns API readiness and whether a model file is present. This does not load the model into memory.

## `GET /model-info`

Returns the served estimator, `serving_candidate`, active `features`, `evaluation_model`, and overall and per-dataset holdout metrics. For a multi-source run, `source_selected` metrics evaluate the full model-selection procedure. `feature_importance_method` identifies tree impurity fractions or increases in training-sample MAE (kcal) after input permutation.

## `POST /calories`

The response includes `estimated_calories` and `model_inputs`, the raw fields used by the loaded model. This release returns `["duration"]`. The five request fields are still required and can be recorded in workout history.

Request JSON:

```json
{
  "age": 25,
  "gender": "Male",
  "weight": 70,
  "duration": 45,
  "heart_rate": 130,
  "user_name": "Alex"
}
```

`user_name` is optional and is used to associate a workout with a name in the optional history store. Numeric ranges are validated by the API. Successful predictions return `200`; invalid payloads return `400`; a missing model returns `503`.

## `GET /workouts?user_name=Alex`

Returns saved workout history if `DATABASE_ENABLED=true`. SQLite can store history locally without a database server; MySQL is also supported when `DATABASE_BACKEND=mysql` and its schema and credentials are configured. Database history is optional and does not block predictions.

## `GET /history?user_name=Alex`

Returns `goals`, `checkins`, `foods`, and `workouts` as separate arrays. Each named record includes `profile_id`, the existing stable `users.user_id`, plus its own log/entry ID. The optional name filter limits all four arrays to the named profile; without a filter all profiles and unnamed workouts are included. Unnamed workouts have a null profile ID. Food entries created before a goal exists still belong to their user ID. Food nutrition estimates and manually entered check-in values are separate and are not added together.

Every goal profile save adds a snapshot to `goal_profile_log`. When the database is first updated, one snapshot of each existing current goal is added; earlier overwritten goal versions cannot be recovered. Check-ins still update the existing entry for a profile and date. The profile ID identifies a person, not a particular goal revision, so older entries are not assigned to a specific goal change.

## `GET /fitness-profile?user_name=Alex`

Returns the saved goal profile for that local profile name. The `profile` value is `null` when the name has no saved goal profile. Requires database storage to be enabled.

## `PUT /fitness-profile`

Creates or updates a goal profile. The target weight must be below the starting weight for `lose_weight`, or above the starting weight for `gain_weight`.

```json
{
  "user_name": "Alex",
  "goal": "lose_weight",
  "gender": "prefer_not_to_say",
  "age": 25,
  "height_cm": 170,
  "starting_weight_kg": 80,
  "target_weight_kg": 72
}
```

Accepted gender values: `female`, `male`, `other`, `prefer_not_to_say`.

## `GET /progress?user_name=Alex`

Returns up to 730 dated progress entries, newest first, when database storage is enabled. A goal profile must exist before the API can save a check-in.

## `POST /progress`

Saves one daily check-in. `user_name`, `entry_date` (`YYYY-MM-DD`), and `weight_kg` are required. The other fields are optional: `workout_duration_minutes`, `calories_burned`, `calories_consumed`, `protein_grams`, `pushups`, and `situps`. A second entry for the same profile and date replaces that day's values. This endpoint stores user-entered information; it does not generate nutrition or goal-time recommendations.

The profile name is a local identifier and is not protected by authentication.

## `GET /foods?q=chicken`

Searches the local FoodData Central survey CSV files. Queries shorter than two characters return an empty list. Results include `fdc_id`, description, calorie and protein values per 100 g, and available gram weights for portions. The API returns `503` when the required local food tables cannot be found or read.

## `POST /food-entries`

Estimates nutrition from the server's local catalog and saves the result for a profile/date. Send `user_name`, `entry_date` (`YYYY-MM-DD`), `fdc_id`, and `amount_g`. The API calculates calories and protein itself; client-supplied nutrition totals are not accepted. Database storage must be enabled.

```json
{
  "user_name": "Alex",
  "entry_date": "2026-11-14",
  "fdc_id": 2705384,
  "amount_g": 200
}
```

The result contains estimated kcal and protein grams. Values are scaled from the food's per-100-g nutrient values using the entered grams. Catalog values are averages and may differ from a prepared meal.

## `GET /food-entries?user_name=Alex&entry_date=2026-11-14`

Returns that date's saved foods and summed estimated kcal/protein. If `entry_date` is omitted, the current date is used. Requires database storage to be enabled.

## `DELETE /food-entries/<entry_id>?user_name=Alex`

Removes one food log item belonging to that local profile. Requires database storage to be enabled.

The FoodData Central data is a reference lookup, not training data for workout calorie or goal-duration models. It does not supply personalized nutrition targets or a prediction of time to reach a weight goal.
