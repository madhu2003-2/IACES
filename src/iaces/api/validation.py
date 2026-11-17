"""Input validation for calorie prediction requests."""

from __future__ import annotations

from datetime import date
from numbers import Real


FIELD_RULES = {
    "age": (10, 100),
    "weight": (20, 350),
    "duration": (1, 300),
    "heart_rate": (30, 240),
}
REQUIRED_FIELDS = ("age", "gender", "weight", "duration", "heart_rate")


def validate_prediction_payload(payload: object) -> tuple[dict[str, object] | None, list[str]]:
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]

    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        errors.extend(f"Missing required field: {field}." for field in missing)

    cleaned: dict[str, object] = {}
    gender = payload.get("gender")
    if isinstance(gender, str):
        normalized_gender = gender.strip().lower()
        if normalized_gender not in {"male", "female"}:
            errors.append("gender must be 'Male' or 'Female'.")
        else:
            cleaned["gender"] = normalized_gender
    elif "gender" in payload:
        errors.append("gender must be a string: 'Male' or 'Female'.")

    for field, (minimum, maximum) in FIELD_RULES.items():
        value = payload.get(field)
        if field not in payload:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            errors.append(f"{field} must be a number.")
            continue
        number = float(value)
        if not minimum <= number <= maximum:
            errors.append(f"{field} must be between {minimum} and {maximum}.")
            continue
        if field == "age" and not number.is_integer():
            errors.append("age must be a whole number of years.")
            continue
        cleaned[field] = int(number) if field == "age" else number

    user_name = payload.get("user_name")
    if user_name is not None:
        if not isinstance(user_name, str) or len(user_name.strip()) > 100:
            errors.append("user_name must be a string of at most 100 characters.")
        elif user_name.strip():
            cleaned["user_name"] = user_name.strip()

    return (None, errors) if errors else (cleaned, [])


def _clean_user_name(payload: dict[str, object], errors: list[str]) -> str | None:
    user_name = payload.get("user_name")
    if not isinstance(user_name, str) or not user_name.strip():
        errors.append("user_name is required and must be a non-empty string.")
        return None
    if len(user_name.strip()) > 100:
        errors.append("user_name must be at most 100 characters.")
        return None
    return user_name.strip()


def _clean_number(
    payload: dict[str, object],
    field: str,
    errors: list[str],
    *,
    minimum: float,
    maximum: float,
    required: bool,
    whole_number: bool = False,
) -> int | float | None:
    if field not in payload or payload[field] is None:
        if required:
            errors.append(f"{field} is required.")
        return None
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append(f"{field} must be a number.")
        return None
    number = float(value)
    if not minimum <= number <= maximum:
        errors.append(f"{field} must be between {minimum:g} and {maximum:g}.")
        return None
    if whole_number and not number.is_integer():
        errors.append(f"{field} must be a whole number.")
        return None
    return int(number) if whole_number else number


def validate_fitness_profile_payload(
    payload: object,
) -> tuple[dict[str, object] | None, list[str]]:
    """Validate the goal profile used by the local progress tracker."""
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]

    errors: list[str] = []
    cleaned: dict[str, object] = {}
    user_name = _clean_user_name(payload, errors)
    if user_name is not None:
        cleaned["user_name"] = user_name

    goal = payload.get("goal")
    if not isinstance(goal, str) or goal.strip().lower() not in {"lose_weight", "gain_weight"}:
        errors.append("goal must be 'lose_weight' or 'gain_weight'.")
    else:
        cleaned["goal"] = goal.strip().lower()

    gender = payload.get("gender")
    allowed_genders = {"female", "male", "other", "prefer_not_to_say"}
    if not isinstance(gender, str) or gender.strip().lower() not in allowed_genders:
        errors.append("gender must be Female, Male, Other, or Prefer not to say.")
    else:
        cleaned["gender"] = gender.strip().lower()

    numeric_fields = (
        ("age", 13, 100, True),
        ("height_cm", 100, 250, False),
        ("starting_weight_kg", 20, 350, False),
        ("target_weight_kg", 20, 350, False),
    )
    for field, minimum, maximum, whole_number in numeric_fields:
        value = _clean_number(
            payload,
            field,
            errors,
            minimum=minimum,
            maximum=maximum,
            required=True,
            whole_number=whole_number,
        )
        if value is not None:
            cleaned[field] = value

    if "goal" in cleaned and "starting_weight_kg" in cleaned and "target_weight_kg" in cleaned:
        start = float(cleaned["starting_weight_kg"])
        target = float(cleaned["target_weight_kg"])
        if cleaned["goal"] == "lose_weight" and target >= start:
            errors.append("For a weight-loss goal, target weight must be below starting weight.")
        if cleaned["goal"] == "gain_weight" and target <= start:
            errors.append("For a weight-gain goal, target weight must be above starting weight.")

    return (None, errors) if errors else (cleaned, [])


def validate_progress_payload(
    payload: object,
) -> tuple[dict[str, object] | None, list[str]]:
    """Validate one dated weight, workout, nutrition, and strength check-in."""
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]

    errors: list[str] = []
    cleaned: dict[str, object] = {}
    user_name = _clean_user_name(payload, errors)
    if user_name is not None:
        cleaned["user_name"] = user_name

    entry_date = payload.get("entry_date")
    if not isinstance(entry_date, str):
        errors.append("entry_date must be a date in YYYY-MM-DD format.")
    else:
        try:
            cleaned["entry_date"] = date.fromisoformat(entry_date.strip()).isoformat()
        except ValueError:
            errors.append("entry_date must be a date in YYYY-MM-DD format.")

    weight = _clean_number(
        payload,
        "weight_kg",
        errors,
        minimum=20,
        maximum=350,
        required=True,
    )
    if weight is not None:
        cleaned["weight_kg"] = weight

    optional_fields = (
        ("workout_duration_minutes", 0, 600, False),
        ("calories_burned", 0, 5000, False),
        ("calories_consumed", 0, 15000, False),
        ("protein_grams", 0, 500, False),
        ("pushups", 0, 5000, True),
        ("situps", 0, 5000, True),
    )
    for field, minimum, maximum, whole_number in optional_fields:
        value = _clean_number(
            payload,
            field,
            errors,
            minimum=minimum,
            maximum=maximum,
            required=False,
            whole_number=whole_number,
        )
        cleaned[field] = value

    return (None, errors) if errors else (cleaned, [])


def validate_food_entry_payload(
    payload: object,
) -> tuple[dict[str, object] | None, list[str]]:
    """Validate a food-log request; nutrition amounts are calculated by the API."""
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]

    errors: list[str] = []
    cleaned: dict[str, object] = {}
    user_name = _clean_user_name(payload, errors)
    if user_name is not None:
        cleaned["user_name"] = user_name

    entry_date = payload.get("entry_date")
    if not isinstance(entry_date, str):
        errors.append("entry_date must be a date in YYYY-MM-DD format.")
    else:
        try:
            cleaned["entry_date"] = date.fromisoformat(entry_date.strip()).isoformat()
        except ValueError:
            errors.append("entry_date must be a date in YYYY-MM-DD format.")

    fdc_id = _clean_number(
        payload,
        "fdc_id",
        errors,
        minimum=1,
        maximum=2_000_000_000,
        required=True,
        whole_number=True,
    )
    if fdc_id is not None:
        cleaned["fdc_id"] = fdc_id

    amount_g = _clean_number(
        payload,
        "amount_g",
        errors,
        minimum=0.1,
        maximum=5000,
        required=True,
    )
    if amount_g is not None:
        cleaned["amount_g"] = amount_g

    return (None, errors) if errors else (cleaned, [])
