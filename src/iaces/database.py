"""Optional SQLite or MySQL persistence for user names and workout predictions."""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import mysql.connector
from flask import g, has_request_context

from iaces.config import DATABASE_BACKEND, DATABASE_ENABLED, DATABASE_PATH


class DatabaseDisabledError(RuntimeError):
    """Raised when database features have not been enabled in .env."""


def _connect_sqlite() -> sqlite3.Connection:
    path = getattr(g, "workout_database_path", DATABASE_PATH) if has_request_context() else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=4)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS workout_sessions (
            workout_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NULL,
            age INTEGER NOT NULL CHECK (age BETWEEN 10 AND 100),
            gender TEXT NOT NULL CHECK (LOWER(gender) IN ('male', 'female')),
            weight_kg REAL NOT NULL CHECK (weight_kg BETWEEN 20 AND 350),
            duration_minutes REAL NOT NULL CHECK (duration_minutes BETWEEN 1 AND 300),
            heart_rate_bpm INTEGER NOT NULL CHECK (heart_rate_bpm BETWEEN 30 AND 240),
            estimated_calories REAL NOT NULL CHECK (estimated_calories >= 0),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_workout_sessions_created_at
            ON workout_sessions(created_at);
        CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_created
            ON workout_sessions(user_id, created_at);

        CREATE TABLE IF NOT EXISTS fitness_profiles (
            user_id INTEGER PRIMARY KEY,
            goal TEXT NOT NULL CHECK (goal IN ('lose_weight', 'gain_weight')),
            gender TEXT NOT NULL CHECK (gender IN ('female', 'male', 'other', 'prefer_not_to_say')),
            age INTEGER NOT NULL CHECK (age BETWEEN 13 AND 100),
            height_cm REAL NOT NULL CHECK (height_cm BETWEEN 100 AND 250),
            starting_weight_kg REAL NOT NULL CHECK (starting_weight_kg BETWEEN 20 AND 350),
            target_weight_kg REAL NOT NULL CHECK (target_weight_kg BETWEEN 20 AND 350),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS progress_entries (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_date TEXT NOT NULL,
            weight_kg REAL NOT NULL CHECK (weight_kg BETWEEN 20 AND 350),
            workout_duration_minutes REAL NULL CHECK (workout_duration_minutes BETWEEN 0 AND 600),
            calories_burned REAL NULL CHECK (calories_burned BETWEEN 0 AND 5000),
            calories_consumed REAL NULL CHECK (calories_consumed BETWEEN 0 AND 15000),
            protein_grams REAL NULL CHECK (protein_grams BETWEEN 0 AND 500),
            pushups INTEGER NULL CHECK (pushups BETWEEN 0 AND 5000),
            situps INTEGER NULL CHECK (situps BETWEEN 0 AND 5000),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_id, entry_date),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS goal_profile_log (
            goal_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            legacy_user_id INTEGER UNIQUE,
            goal TEXT NOT NULL,
            gender TEXT NOT NULL,
            age INTEGER NOT NULL,
            height_cm REAL NOT NULL,
            starting_weight_kg REAL NOT NULL,
            target_weight_kg REAL NOT NULL,
            saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_goal_profile_log_user ON goal_profile_log(user_id, saved_at);

        INSERT OR IGNORE INTO goal_profile_log
            (user_id, legacy_user_id, goal, gender, age, height_cm, starting_weight_kg, target_weight_kg, saved_at)
        SELECT fp.user_id, fp.user_id, fp.goal, fp.gender, fp.age, fp.height_cm,
               fp.starting_weight_kg, fp.target_weight_kg, fp.updated_at
        FROM fitness_profiles AS fp
        WHERE NOT EXISTS (SELECT 1 FROM goal_profile_log AS gl WHERE gl.user_id = fp.user_id);

        CREATE TABLE IF NOT EXISTS food_entries (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_date TEXT NOT NULL,
            fdc_id INTEGER NOT NULL,
            food_description TEXT NOT NULL,
            amount_g REAL NOT NULL CHECK (amount_g > 0 AND amount_g <= 5000),
            calories_kcal REAL NOT NULL CHECK (calories_kcal >= 0),
            protein_g REAL NOT NULL CHECK (protein_g >= 0),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_food_entries_user_date
            ON food_entries(user_id, entry_date);

        """
    )
    return connection


def _connect():
    if not DATABASE_ENABLED:
        raise DatabaseDisabledError("Set DATABASE_ENABLED=true in .env to enable history.")
    if DATABASE_BACKEND == "sqlite":
        return _connect_sqlite()
    if DATABASE_BACKEND == "mysql":
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "fitpredict_user"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "fitpredict_ai"),
            connection_timeout=4,
        )

    raise ValueError("DATABASE_BACKEND must be either 'sqlite' or 'mysql'.")


def _ensure_mysql_goal_log(connection) -> None:
    """Create and backfill goal history when using the new MySQL features."""
    if DATABASE_BACKEND != "mysql":
        return
    cursor = connection.cursor()
    try:
        try:
            cursor.execute("SELECT goal_log_id FROM goal_profile_log LIMIT 0")
        except mysql.connector.Error as exc:
            if exc.errno != 1146:
                raise
            cursor.execute("""CREATE TABLE IF NOT EXISTS goal_profile_log (
                    goal_log_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT UNSIGNED NOT NULL,
                    legacy_user_id BIGINT UNSIGNED NULL UNIQUE,
                    goal VARCHAR(20) NOT NULL, gender VARCHAR(20) NOT NULL,
                    age SMALLINT NOT NULL, height_cm DECIMAL(6,2) NOT NULL,
                    starting_weight_kg DECIMAL(6,2) NOT NULL, target_weight_kg DECIMAL(6,2) NOT NULL,
                    saved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    KEY idx_goal_log_user (user_id, saved_at),
                    CONSTRAINT fk_goal_log_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )""")
        cursor.execute("""INSERT IGNORE INTO goal_profile_log
                    (user_id, legacy_user_id, goal, gender, age, height_cm, starting_weight_kg, target_weight_kg, saved_at)
                    SELECT fp.user_id, fp.user_id, fp.goal, fp.gender, fp.age, fp.height_cm,
                           fp.starting_weight_kg, fp.target_weight_kg, fp.updated_at
                    FROM fitness_profiles AS fp
                    WHERE NOT EXISTS (SELECT 1 FROM goal_profile_log AS gl WHERE gl.user_id = fp.user_id)""")
        connection.commit()
    finally:
        cursor.close()


def save_workout(values: dict[str, object], calories: float) -> None:
    connection = _connect()
    cursor = connection.cursor()
    try:
        user_id = None
        user_name = values.get("user_name")
        if user_name:
            if DATABASE_BACKEND == "sqlite":
                cursor.execute(
                    "SELECT user_id FROM users WHERE display_name = ?",
                    (user_name,),
                )
            else:
                cursor.execute(
                    "SELECT user_id FROM users WHERE display_name = %s",
                    (user_name,),
                )
            row = cursor.fetchone()
            if row:
                user_id = row[0]
            else:
                placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
                cursor.execute(
                    f"INSERT INTO users (display_name) VALUES ({placeholder})",
                    (user_name,),
                )
                user_id = cursor.lastrowid

        placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
        cursor.execute(
            """INSERT INTO workout_sessions
               (user_id, age, gender, weight_kg, duration_minutes, heart_rate_bpm,
                estimated_calories)
               VALUES (""" + ", ".join([placeholder] * 7) + ")",
            (
                user_id,
                values["age"],
                values["gender"],
                values["weight"],
                values["duration"],
                values["heart_rate"],
                calories,
            ),
        )
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def get_workouts(user_name: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    connection = _connect()
    cursor = (
        connection.cursor()
        if DATABASE_BACKEND == "sqlite"
        else connection.cursor(dictionary=True)
    )
    try:
        query = """SELECT ws.workout_id, u.display_name AS user_name, ws.age,
                          ws.gender, ws.weight_kg, ws.duration_minutes,
                          ws.heart_rate_bpm, ws.estimated_calories, ws.created_at
                   FROM workout_sessions AS ws
                   LEFT JOIN users AS u ON u.user_id = ws.user_id"""
        parameters: list[object] = []
        if user_name:
            query += " WHERE u.display_name = ?" if DATABASE_BACKEND == "sqlite" else " WHERE u.display_name = %s"
            parameters.append(user_name)
        query += " ORDER BY ws.created_at DESC LIMIT ?" if DATABASE_BACKEND == "sqlite" else " ORDER BY ws.created_at DESC LIMIT %s"
        parameters.append(limit)
        cursor.execute(query, tuple(parameters))
        rows = cursor.fetchall()
        if DATABASE_BACKEND == "sqlite":
            return [dict(row) for row in rows]
        result = []
        for row in rows:
            if row.get("created_at") is not None:
                row["created_at"] = row["created_at"].isoformat()
            result.append(row)
        return result
    finally:
        cursor.close()
        connection.close()


def _ensure_user_id(cursor, user_name: str) -> int:
    placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
    cursor.execute(
        f"SELECT user_id FROM users WHERE display_name = {placeholder}",
        (user_name,),
    )
    row = cursor.fetchone()
    if row:
        return int(row[0])

    if DATABASE_BACKEND == "sqlite":
        cursor.execute(
            "INSERT INTO users (display_name) VALUES (?) ON CONFLICT(display_name) DO NOTHING",
            (user_name,),
        )
    else:
        cursor.execute("INSERT IGNORE INTO users (display_name) VALUES (%s)", (user_name,))
    cursor.execute(
        f"SELECT user_id FROM users WHERE display_name = {placeholder}",
        (user_name,),
    )
    row = cursor.fetchone()
    if not row:
        raise RuntimeError("The local profile could not be created.")
    return int(row[0])


def _serialize_database_row(row: Any) -> dict[str, Any]:
    values = dict(row)
    for key, value in values.items():
        if isinstance(value, Decimal):
            values[key] = float(value)
        elif isinstance(value, (date, datetime)):
            values[key] = value.isoformat()
    return values


def save_fitness_profile(values: dict[str, object]) -> dict[str, Any]:
    """Create or update the local user's goal and baseline measurements."""
    connection = _connect()
    try:
        _ensure_mysql_goal_log(connection)
    except Exception:
        connection.close()
        raise
    cursor = connection.cursor()
    try:
        user_name = str(values["user_name"])
        user_id = _ensure_user_id(cursor, user_name)
        columns = (
            "goal", "gender", "age", "height_cm", "starting_weight_kg", "target_weight_kg"
        )
        placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
        placeholders = ", ".join([placeholder] * len(columns))
        field_list = ", ".join(columns)
        update_list = ", ".join(f"{column} = excluded.{column}" for column in columns)
        if DATABASE_BACKEND == "sqlite":
            sql = (
                f"INSERT INTO fitness_profiles (user_id, {field_list}, created_at, updated_at) "
                f"VALUES (?, {placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                f"ON CONFLICT(user_id) DO UPDATE SET {update_list}, updated_at = CURRENT_TIMESTAMP"
            )
        else:
            mysql_updates = ", ".join(f"{column} = VALUES({column})" for column in columns)
            sql = (
                f"INSERT INTO fitness_profiles (user_id, {field_list}) "
                f"VALUES (%s, {placeholders}) ON DUPLICATE KEY UPDATE "
                f"{mysql_updates}, updated_at = CURRENT_TIMESTAMP"
            )
        cursor.execute(
            sql,
            (user_id, *(values[column] for column in columns)),
        )
        cursor.execute(
            f"INSERT INTO goal_profile_log (user_id, {field_list}) VALUES ({placeholder}, {placeholders})",
            (user_id, *(values[column] for column in columns)),
        )
        connection.commit()
        return {**values, "profile_id": user_id}
    finally:
        cursor.close()
        connection.close()


def get_fitness_profile(user_name: str) -> dict[str, Any] | None:
    connection = _connect()
    cursor = (
        connection.cursor()
        if DATABASE_BACKEND == "sqlite"
        else connection.cursor(dictionary=True)
    )
    placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
    try:
        cursor.execute(
            """SELECT u.user_id AS profile_id, u.display_name AS user_name, fp.goal, fp.gender, fp.age,
                      fp.height_cm, fp.starting_weight_kg, fp.target_weight_kg,
                      fp.created_at, fp.updated_at
               FROM fitness_profiles AS fp
               JOIN users AS u ON u.user_id = fp.user_id
               WHERE u.display_name = """ + placeholder,
            (user_name,),
        )
        row = cursor.fetchone()
        return _serialize_database_row(row) if row else None
    finally:
        cursor.close()
        connection.close()


def save_progress_entry(values: dict[str, object]) -> dict[str, Any]:
    """Save one daily check-in; a second save for that date replaces the entry."""
    connection = _connect()
    cursor = connection.cursor()
    try:
        user_name = str(values["user_name"])
        placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
        cursor.execute(
            """SELECT fp.user_id FROM fitness_profiles AS fp
               JOIN users AS u ON u.user_id = fp.user_id
               WHERE u.display_name = """ + placeholder,
            (user_name,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError("Save a goal profile before adding a progress check-in.")
        user_id = int(row[0])

        columns = (
            "entry_date", "weight_kg", "workout_duration_minutes", "calories_burned",
            "calories_consumed", "protein_grams", "pushups", "situps",
        )
        placeholders = ", ".join([placeholder] * len(columns))
        if DATABASE_BACKEND == "sqlite":
            updates = ", ".join(f"{column} = excluded.{column}" for column in columns)
            sql = (
                f"INSERT INTO progress_entries (user_id, {', '.join(columns)}, created_at, updated_at) "
                f"VALUES ({placeholder}, {placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                f"ON CONFLICT(user_id, entry_date) DO UPDATE SET "
                f"{updates}, updated_at = CURRENT_TIMESTAMP"
            )
        else:
            updates = ", ".join(f"{column} = VALUES({column})" for column in columns)
            sql = (
                f"INSERT INTO progress_entries (user_id, {', '.join(columns)}) "
                f"VALUES ({placeholder}, {placeholders}) ON DUPLICATE KEY UPDATE "
                f"{updates}, updated_at = CURRENT_TIMESTAMP"
            )
        cursor.execute(sql, (user_id, *(values[column] for column in columns)))
        connection.commit()
        return {key: value for key, value in values.items() if key != "user_name"} | {
            "user_name": user_name
        }
    finally:
        cursor.close()
        connection.close()


def get_progress_entries(user_name: str, limit: int = 730) -> list[dict[str, Any]]:
    connection = _connect()
    cursor = (
        connection.cursor()
        if DATABASE_BACKEND == "sqlite"
        else connection.cursor(dictionary=True)
    )
    placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
    try:
        cursor.execute(
            """SELECT pe.entry_date, pe.weight_kg, pe.workout_duration_minutes,
                      pe.calories_burned, pe.calories_consumed, pe.protein_grams,
                      pe.pushups, pe.situps, pe.created_at, pe.updated_at
               FROM progress_entries AS pe
               JOIN users AS u ON u.user_id = pe.user_id
               WHERE u.display_name = """ + placeholder + " ORDER BY pe.entry_date DESC LIMIT " + placeholder,
            (user_name, limit),
        )
        return [_serialize_database_row(row) for row in cursor.fetchall()]
    finally:
        cursor.close()
        connection.close()


def save_food_entry(values: dict[str, object]) -> dict[str, Any]:
    """Save one food and its server-calculated nutrition estimate."""
    connection = _connect()
    cursor = connection.cursor()
    try:
        user_id = _ensure_user_id(cursor, str(values["user_name"]))
        placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
        cursor.execute(
            "INSERT INTO food_entries "
            "(user_id, entry_date, fdc_id, food_description, amount_g, calories_kcal, protein_g) "
            f"VALUES ({', '.join([placeholder] * 7)})",
            (
                user_id,
                values["entry_date"],
                values["fdc_id"],
                values["food_description"],
                values["amount_g"],
                values["calories_kcal"],
                values["protein_g"],
            ),
        )
        connection.commit()
        return {key: value for key, value in values.items() if key != "user_name"}
    finally:
        cursor.close()
        connection.close()


def get_food_entries(user_name: str, entry_date: str) -> list[dict[str, Any]]:
    """Return one user's food log for a calendar date, newest item first."""
    connection = _connect()
    cursor = (
        connection.cursor()
        if DATABASE_BACKEND == "sqlite"
        else connection.cursor(dictionary=True)
    )
    placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
    try:
        cursor.execute(
            "SELECT fe.entry_id, fe.entry_date, fe.fdc_id, fe.food_description, "
            "fe.amount_g, fe.calories_kcal, fe.protein_g, fe.created_at "
            "FROM food_entries AS fe JOIN users AS u ON u.user_id = fe.user_id "
            f"WHERE u.display_name = {placeholder} AND fe.entry_date = {placeholder} "
            "ORDER BY fe.entry_id DESC",
            (user_name, entry_date),
        )
        return [_serialize_database_row(row) for row in cursor.fetchall()]
    finally:
        cursor.close()
        connection.close()


def delete_food_entry(user_name: str, entry_id: int) -> bool:
    """Delete a food-log item only when it belongs to the named local profile."""
    connection = _connect()
    cursor = connection.cursor()
    placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
    try:
        cursor.execute(
            f"SELECT user_id FROM users WHERE display_name = {placeholder}",
            (user_name,),
        )
        row = cursor.fetchone()
        if not row:
            return False
        cursor.execute(
            f"DELETE FROM food_entries WHERE entry_id = {placeholder} AND user_id = {placeholder}",
            (entry_id, row[0]),
        )
        deleted = cursor.rowcount > 0
        connection.commit()
        return deleted
    finally:
        cursor.close()
        connection.close()


def get_linked_history(user_name: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return separate logs linked by their stable user/profile ID, without multiplying rows."""
    connection = _connect()
    try:
        _ensure_mysql_goal_log(connection)
    except Exception:
        connection.close()
        raise
    cursor = connection.cursor() if DATABASE_BACKEND == "sqlite" else connection.cursor(dictionary=True)
    placeholder = "?" if DATABASE_BACKEND == "sqlite" else "%s"
    filters = " WHERE u.display_name = " + placeholder if user_name else ""
    params = (user_name,) if user_name else ()
    queries = {
        "goals": """SELECT gl.goal_log_id, u.user_id AS profile_id, u.display_name AS user_name,
                    gl.goal, gl.gender, gl.age, gl.height_cm, gl.starting_weight_kg,
                    gl.target_weight_kg, gl.saved_at
                    FROM goal_profile_log gl JOIN users u ON u.user_id = gl.user_id""",
        "checkins": """SELECT pe.entry_id, u.user_id AS profile_id, u.display_name AS user_name,
                       pe.entry_date, pe.weight_kg, pe.workout_duration_minutes, pe.calories_burned,
                       pe.calories_consumed, pe.protein_grams, pe.pushups, pe.situps,
                       pe.created_at, pe.updated_at
                       FROM progress_entries pe JOIN users u ON u.user_id = pe.user_id""",
        "foods": """SELECT fe.entry_id, u.user_id AS profile_id, u.display_name AS user_name,
                    fe.entry_date, fe.fdc_id, fe.food_description, fe.amount_g, fe.calories_kcal,
                    fe.protein_g, fe.created_at
                    FROM food_entries fe JOIN users u ON u.user_id = fe.user_id""",
        "workouts": """SELECT ws.workout_id, u.user_id AS profile_id, u.display_name AS user_name,
                       ws.age, ws.gender, ws.weight_kg, ws.duration_minutes, ws.heart_rate_bpm,
                       ws.estimated_calories, ws.created_at
                       FROM workout_sessions ws LEFT JOIN users u ON u.user_id = ws.user_id""",
    }
    order = {"goals": "gl.saved_at DESC, gl.goal_log_id DESC", "checkins": "pe.entry_date DESC, pe.entry_id DESC",
             "foods": "fe.entry_date DESC, fe.entry_id DESC", "workouts": "ws.created_at DESC, ws.workout_id DESC"}
    try:
        result = {}
        for key, query in queries.items():
            cursor.execute(query + filters + " ORDER BY " + order[key], params)
            result[key] = [_serialize_database_row(row) for row in cursor.fetchall()]
        return result
    finally:
        cursor.close()
        connection.close()
