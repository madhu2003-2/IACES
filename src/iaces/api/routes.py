"""HTTP routes for health, calorie prediction, and workout history."""

from __future__ import annotations

import logging
from datetime import date

from flask import Blueprint, jsonify, request

from iaces.config import DATABASE_ENABLED, MODEL_PATH
from iaces.database import (
    get_fitness_profile,
    get_linked_history,
    delete_food_entry,
    get_food_entries,
    get_progress_entries,
    get_workouts,
    save_fitness_profile,
    save_food_entry,
    save_progress_entry,
    save_workout,
)
from iaces.api.services import calorie_model_inputs, describe_model, estimate_calories
from iaces.api.validation import (
    validate_fitness_profile_payload,
    validate_food_entry_payload,
    validate_prediction_payload,
    validate_progress_payload,
)
from iaces.ml.predict import ModelUnavailableError
from iaces.ml.fitness_assessment import AssessmentModelUnavailable, predict_assessment, validate as validate_assessment
from iaces.nutrition import NutritionDataUnavailableError, estimate_food, search_foods


LOGGER = logging.getLogger("fitpredict.api")
api = Blueprint("api", __name__)


@api.post("/fitness-assessment")
def fitness_assessment():
    values, errors = validate_assessment(request.get_json(silent=True))
    if errors:
        return jsonify({"success": False, "errors": errors}), 400
    try:
        result = predict_assessment(values)
    except AssessmentModelUnavailable as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception:
        LOGGER.exception("Fitness assessment prediction failed")
        return jsonify({"success": False, "error": "Fitness assessment is unavailable."}), 500
    return jsonify({"success": True, **result}), 200


@api.get("/health")
def health():
    model_available = MODEL_PATH.is_file()
    return jsonify(
        {
            "success": True,
            "status": "ok",
            "model_available": model_available,
            "database_enabled": DATABASE_ENABLED,
        }
    ), 200


@api.get("/model-info")
def model_info():
    try:
        return jsonify({"success": True, **describe_model()}), 200
    except ModelUnavailableError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception:
        LOGGER.exception("Model information could not be loaded")
        return jsonify({"success": False, "error": "Model information is unavailable."}), 500


@api.post("/calories")
def calories():
    payload = request.get_json(silent=True)
    values, errors = validate_prediction_payload(payload)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    try:
        estimate = estimate_calories(values)
        model_inputs = calorie_model_inputs()
    except ModelUnavailableError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception:
        LOGGER.exception("Prediction failed")
        return jsonify({"success": False, "error": "Prediction failed due to a server error."}), 500

    history_saved = False
    history_message = "Workout history is disabled."
    if DATABASE_ENABLED:
        try:
            save_workout(values, estimate)
            history_saved = True
            history_message = "Workout saved."
        except Exception:
            LOGGER.exception("Prediction succeeded but workout history could not be saved")
            history_message = "Prediction succeeded, but workout history could not be saved."

    return jsonify(
        {
            "success": True,
            "estimated_calories": round(estimate, 2),
            "model_inputs": model_inputs,
            "unit": "kcal",
            "history_saved": history_saved,
            "history_message": history_message,
        }
    ), 200


@api.get("/workouts")
def workouts():
    if not DATABASE_ENABLED:
        return jsonify(
            {
                "success": False,
                "error": "Workout history is disabled. Set DATABASE_ENABLED=true in .env and choose SQLite or MySQL.",
            }
        ), 503

    user_name = request.args.get("user_name", "").strip() or None
    try:
        rows = get_workouts(user_name=user_name)
    except Exception:
        LOGGER.exception("Workout history could not be retrieved")
        return jsonify({"success": False, "error": "Workout history is unavailable."}), 503
    return jsonify({"success": True, "workouts": rows}), 200


@api.get("/history")
def linked_history():
    if not DATABASE_ENABLED:
        return jsonify({"success": False, "error": "History is disabled. Enable SQLite or MySQL in .env."}), 503
    user_name = request.args.get("user_name", "").strip() or None
    try:
        logs = get_linked_history(user_name=user_name)
    except Exception:
        LOGGER.exception("Linked history could not be retrieved")
        return jsonify({"success": False, "error": "History is unavailable."}), 503
    return jsonify({"success": True, **logs}), 200


@api.get("/fitness-profile")
def fitness_profile():
    if not DATABASE_ENABLED:
        return jsonify(
            {"success": False, "error": "Progress tracking is disabled. Enable SQLite or MySQL in .env."}
        ), 503
    user_name = request.args.get("user_name", "").strip()
    if not user_name:
        return jsonify({"success": False, "error": "user_name is required."}), 400
    try:
        profile = get_fitness_profile(user_name)
    except Exception:
        LOGGER.exception("Fitness profile could not be retrieved")
        return jsonify({"success": False, "error": "Fitness profile is unavailable."}), 503
    return jsonify({"success": True, "profile": profile}), 200


@api.put("/fitness-profile")
def save_fitness_profile_route():
    if not DATABASE_ENABLED:
        return jsonify(
            {"success": False, "error": "Progress tracking is disabled. Enable SQLite or MySQL in .env."}
        ), 503
    values, errors = validate_fitness_profile_payload(request.get_json(silent=True))
    if errors:
        return jsonify({"success": False, "errors": errors}), 400
    try:
        profile = save_fitness_profile(values)
    except Exception:
        LOGGER.exception("Fitness profile could not be saved")
        return jsonify({"success": False, "error": "Fitness profile could not be saved."}), 503
    return jsonify({"success": True, "profile": profile}), 200


@api.get("/progress")
def progress_entries():
    if not DATABASE_ENABLED:
        return jsonify(
            {"success": False, "error": "Progress tracking is disabled. Enable SQLite or MySQL in .env."}
        ), 503
    user_name = request.args.get("user_name", "").strip()
    if not user_name:
        return jsonify({"success": False, "error": "user_name is required."}), 400
    try:
        entries = get_progress_entries(user_name)
    except Exception:
        LOGGER.exception("Progress entries could not be retrieved")
        return jsonify({"success": False, "error": "Progress history is unavailable."}), 503
    return jsonify({"success": True, "entries": entries}), 200


@api.post("/progress")
def save_progress():
    if not DATABASE_ENABLED:
        return jsonify(
            {"success": False, "error": "Progress tracking is disabled. Enable SQLite or MySQL in .env."}
        ), 503
    values, errors = validate_progress_payload(request.get_json(silent=True))
    if errors:
        return jsonify({"success": False, "errors": errors}), 400
    try:
        entry = save_progress_entry(values)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        LOGGER.exception("Progress entry could not be saved")
        return jsonify({"success": False, "error": "Progress entry could not be saved."}), 503
    return jsonify({"success": True, "entry": entry}), 200


@api.get("/foods")
def foods():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify({"success": True, "foods": []}), 200
    try:
        matches = search_foods(query)
    except NutritionDataUnavailableError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception:
        LOGGER.exception("Local nutrition catalog could not be searched")
        return jsonify({"success": False, "error": "The food catalog could not be read."}), 503
    return jsonify({"success": True, "foods": matches}), 200


@api.post("/food-entries")
def add_food_entry():
    if not DATABASE_ENABLED:
        return jsonify(
            {"success": False, "error": "Food logging is disabled. Enable SQLite or MySQL in .env."}
        ), 503
    values, errors = validate_food_entry_payload(request.get_json(silent=True))
    if errors:
        return jsonify({"success": False, "errors": errors}), 400
    try:
        estimate = estimate_food(int(values["fdc_id"]), float(values["amount_g"]))
        saved_values = {
            **values,
            "food_description": estimate["description"],
            "calories_kcal": estimate["estimated_calories_kcal"],
            "protein_g": estimate["estimated_protein_g"],
        }
        entry = save_food_entry(saved_values)
    except NutritionDataUnavailableError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except KeyError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except Exception:
        LOGGER.exception("Food entry could not be saved")
        return jsonify({"success": False, "error": "Food entry could not be saved."}), 503
    return jsonify(
        {
            "success": True,
            "entry": entry,
            "estimate": {
                "calories_kcal": estimate["estimated_calories_kcal"],
                "protein_g": estimate["estimated_protein_g"],
                "amount_g": estimate["amount_g"],
            },
        }
    ), 201


@api.get("/food-entries")
def food_entries():
    if not DATABASE_ENABLED:
        return jsonify(
            {"success": False, "error": "Food logging is disabled. Enable SQLite or MySQL in .env."}
        ), 503
    user_name = request.args.get("user_name", "").strip()
    if not user_name:
        return jsonify({"success": False, "error": "user_name is required."}), 400
    entry_date = request.args.get("entry_date", date.today().isoformat())
    try:
        entry_date = date.fromisoformat(entry_date).isoformat()
    except ValueError:
        return jsonify({"success": False, "error": "entry_date must be YYYY-MM-DD."}), 400
    try:
        entries = get_food_entries(user_name, entry_date)
    except Exception:
        LOGGER.exception("Food entries could not be retrieved")
        return jsonify({"success": False, "error": "Food log is unavailable."}), 503
    return jsonify(
        {
            "success": True,
            "entries": entries,
            "daily_totals": {
                "calories_kcal": round(sum(float(row["calories_kcal"]) for row in entries), 2),
                "protein_g": round(sum(float(row["protein_g"]) for row in entries), 2),
            },
        }
    ), 200


@api.delete("/food-entries/<int:entry_id>")
def remove_food_entry(entry_id: int):
    if not DATABASE_ENABLED:
        return jsonify(
            {"success": False, "error": "Food logging is disabled. Enable SQLite or MySQL in .env."}
        ), 503
    user_name = request.args.get("user_name", "").strip()
    if not user_name:
        return jsonify({"success": False, "error": "user_name is required."}), 400
    try:
        deleted = delete_food_entry(user_name, entry_id)
    except Exception:
        LOGGER.exception("Food entry could not be deleted")
        return jsonify({"success": False, "error": "Food entry could not be deleted."}), 503
    if not deleted:
        return jsonify({"success": False, "error": "Food entry was not found."}), 404
    return jsonify({"success": True}), 200
