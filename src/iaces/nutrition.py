"""Nutrition lookups backed by a local FoodData Central survey export."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from iaces.config import PROJECT_ROOT


TABLES = ("food.csv", "food_nutrient.csv", "food_portion.csv", "nutrient.csv")


class NutritionDataUnavailableError(RuntimeError):
    """Raised when a usable local FoodData Central export is not installed."""


def _find_food_data_directory() -> Path:
    data_root = PROJECT_ROOT / "data"
    for source_root in (data_root / "raw", data_root / "processed"):
        if not source_root.exists():
            continue
        for food_file in source_root.rglob("food.csv"):
            candidate = food_file.parent
            if all((candidate / table).is_file() for table in TABLES):
                return candidate
    raise NutritionDataUnavailableError(
        "FoodData Central files were not found. Place food.csv, food_nutrient.csv, "
        "food_portion.csv, and nutrient.csv together under data/raw or data/processed."
    )


def _as_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _load_food_catalog() -> dict[int, dict[str, Any]]:
    """Read the catalog once, joining nutrient_nbr to food_nutrient.nutrient_id."""
    directory = _find_food_data_directory()

    available_nutrients: set[str] = set()
    with (directory / "nutrient.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            number = (row.get("nutrient_nbr") or "").strip()
            if number in {"203", "208"}:
                available_nutrients.add(number)

    if available_nutrients != {"203", "208"}:
        raise NutritionDataUnavailableError(
            "The FoodData Central nutrient table is missing protein or energy definitions."
        )

    foods: dict[int, dict[str, Any]] = {}
    with (directory / "food.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                fdc_id = int(row["fdc_id"])
            except (KeyError, ValueError):
                continue
            foods[fdc_id] = {
                "fdc_id": fdc_id,
                "description": row.get("description", "").strip(),
                "protein_g_per_100g": None,
                "calories_kcal_per_100g": None,
                "portions": [],
            }

    # In the FNDDS survey export nutrient_id stores nutrient_nbr values, not nutrient.csv.id.
    wanted_nutrient = {
        "203": "protein_g_per_100g",
        "208": "calories_kcal_per_100g",
    }
    with (directory / "food_nutrient.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target = wanted_nutrient.get((row.get("nutrient_id") or "").strip())
            if target is None:
                continue
            try:
                food = foods.get(int(row["fdc_id"]))
            except (KeyError, ValueError):
                continue
            amount = _as_float(row.get("amount"))
            if food is not None and amount is not None:
                food[target] = amount

    with (directory / "food_portion.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                food = foods.get(int(row["fdc_id"]))
            except (KeyError, ValueError):
                continue
            grams = _as_float(row.get("gram_weight"))
            description = (row.get("portion_description") or "").strip()
            if food is not None and grams is not None and grams > 0 and description:
                food["portions"].append(
                    {"description": description, "grams": grams}
                )

    return {
        fdc_id: food
        for fdc_id, food in foods.items()
        if food["protein_g_per_100g"] is not None
        and food["calories_kcal_per_100g"] is not None
    }


def search_foods(query: str, limit: int = 25) -> list[dict[str, Any]]:
    """Find foods by description; only foods with both kcal and protein are included."""
    normalized = query.strip().casefold()
    if len(normalized) < 2:
        return []
    matches = [
        item
        for item in _load_food_catalog().values()
        if normalized in item["description"].casefold()
    ]
    matches.sort(key=lambda item: item["description"].casefold())
    return matches[:limit]


def estimate_food(fdc_id: int, grams: float) -> dict[str, Any]:
    """Estimate kcal and protein from the catalog's per-100-g edible values."""
    food = _load_food_catalog().get(fdc_id)
    if food is None:
        raise KeyError(f"Food ID {fdc_id} was not found in the local catalog.")
    factor = grams / 100.0
    return {
        **food,
        "amount_g": grams,
        "estimated_calories_kcal": round(food["calories_kcal_per_100g"] * factor, 2),
        "estimated_protein_g": round(food["protein_g_per_100g"] * factor, 2),
    }
