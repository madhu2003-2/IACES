"""FitPredict AI dashboard; predictions are always requested from Flask."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="FitPredict AI", page_icon="🏃", layout="wide")

api_url = os.getenv("FITPREDICT_API_URL", "http://127.0.0.1:5000").rstrip("/")
st.title("FitPredict AI")
st.caption("Workout calorie estimates powered by your trained model.")

with st.sidebar:
    st.subheader("Connection")
    api_url = st.text_input("Flask API address", value=api_url).rstrip("/")
    if st.button("Check API"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if health.get("model_available"):
                st.success("API is ready and a model is available.")
            else:
                st.warning("API is running. Train the model before predicting.")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

predict_tab, progress_tab, analytics_tab, history_tab = st.tabs(
    ["Predict calories", "Goal and progress", "Analytics", "Workout history"]
)

with predict_tab:
    st.subheader("Workout details")
    st.write("Enter a few workout details to request an estimate from the Flask API.")
    with st.form("workout_form"):
        user_name = st.text_input("Name for workout history (optional)", max_chars=100)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=10, max_value=100, value=25, step=1)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=350.0, value=70.0, step=0.5)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col2:
            duration = st.slider("Workout duration (minutes)", min_value=1, max_value=300, value=45)
            heart_rate = st.slider("Average heart rate (bpm)", min_value=30, max_value=240, value=130)
        submitted = st.form_submit_button("Predict calories", type="primary")

    if submitted:
        body = {
            "age": int(age),
            "gender": gender,
            "weight": float(weight),
            "duration": int(duration),
            "heart_rate": int(heart_rate),
            "user_name": user_name.strip() or None,
        }
        try:
            response = requests.post(f"{api_url}/calories", json=body, timeout=60)
            result = response.json()
            if response.ok and result.get("success"):
                st.session_state["latest_prediction"] = {
                    "calories": result["estimated_calories"],
                    "model_inputs": result.get("model_inputs"),
                    "duration": duration,
                    "heart_rate": heart_rate,
                    "gender": gender,
                    "weight": weight,
                    "user_name": user_name,
                    "history_saved": result.get("history_saved", False),
                    "history_message": result.get("history_message", ""),
                }
            else:
                st.error(result.get("error") or "; ".join(result.get("errors", [])) or "The API rejected the request.")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        except ValueError:
            st.error("The API returned a response that was not valid JSON.")

    latest = st.session_state.get("latest_prediction")
    if latest:
        st.divider()
        st.metric("Estimated calories burned", f"{latest['calories']:.2f} kcal")
        if latest.get("model_inputs") == ["duration"]:
            st.caption("This model uses workout duration only. Two workouts of the same length receive the same estimate.")
        elif latest.get("model_inputs") == ["duration", "heart_rate"]:
            st.caption("This model uses workout duration and average heart rate.")
        elif latest.get("model_inputs") == []:
            st.caption("This model returns the median calories from its training workouts.")
        st.caption(
            f"{latest['duration']} minutes · {latest['heart_rate']} bpm average heart rate · "
            f"{latest['weight']:.1f} kg"
        )
        st.caption("Demonstration estimate only; not a medical measurement.")
        if latest["history_saved"]:
            st.success(latest["history_message"])
        else:
            st.info(latest["history_message"])


def load_history(name: str = "") -> pd.DataFrame:
    params = {"user_name": name} if name.strip() else None
    response = requests.get(f"{api_url}/workouts", params=params, timeout=8)
    if not response.ok:
        detail = response.json().get("error", "Workout history is unavailable.")
        raise RuntimeError(detail)
    records = response.json().get("workouts", [])
    return pd.DataFrame.from_records(records)


def progress_api(method: str, route: str, **kwargs) -> dict[str, object]:
    response = requests.request(method, f"{api_url}{route}", timeout=10, **kwargs)
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("The API returned a response that was not valid JSON.") from exc
    if not response.ok or not result.get("success"):
        message = result.get("error") or "; ".join(result.get("errors", []))
        raise RuntimeError(message or "The API could not complete the request.")
    return result


with progress_tab:
    st.subheader("Goal and progress tracking")
    st.write(
        "Save a goal profile, then record dated weight, workout, nutrition, and strength check-ins."
    )
    

    fitness_name = st.text_input(
        "Profile name for this local tracker",
        max_chars=100,
        key="fitness_profile_name",
        help="Profiles are saved under this name in the local database.",
    ).strip()
    st.caption("The app has no sign-in; anyone using the same local app can enter this profile name.")
    stored_profile = st.session_state.get("fitness_profile")
    if stored_profile and stored_profile.get("user_name") != fitness_name:
        stored_profile = None

    if st.button("Load saved profile and entries", key="load_fitness_profile"):
        if not fitness_name:
            st.info("Enter your profile name first.")
        else:
            try:
                profile_result = progress_api(
                    "GET", "/fitness-profile", params={"user_name": fitness_name}
                )
                loaded_profile = profile_result.get("profile")
                st.session_state["fitness_profile"] = loaded_profile
                st.session_state["fitness_profile_form_owner"] = None
                if loaded_profile:
                    entry_result = progress_api(
                        "GET", "/progress", params={"user_name": fitness_name}
                    )
                    st.session_state["fitness_entries"] = entry_result.get("entries", [])
                    st.success("Saved profile and progress entries loaded.")
                else:
                    st.session_state["fitness_entries"] = []
                    st.info("No saved profile was found. Fill in the profile below to create one.")
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                st.error(str(exc))

    stored_profile = st.session_state.get("fitness_profile")
    if stored_profile and stored_profile.get("user_name") != fitness_name:
        stored_profile = None
    if st.session_state.get("fitness_profile_form_owner") != fitness_name:
        defaults = {
            "fitness_goal": "lose_weight",
            "fitness_gender": "prefer_not_to_say",
            "fitness_age": 25,
            "fitness_height_cm": 170.0,
            "fitness_start_weight": 70.0,
            "fitness_target_weight": 65.0,
        }
        if stored_profile:
            defaults.update(
                {
                    "fitness_goal": stored_profile["goal"],
                    "fitness_gender": stored_profile["gender"],
                    "fitness_age": int(stored_profile["age"]),
                    "fitness_height_cm": float(stored_profile["height_cm"]),
                    "fitness_start_weight": float(stored_profile["starting_weight_kg"]),
                    "fitness_target_weight": float(stored_profile["target_weight_kg"]),
                }
            )
        st.session_state.update(defaults)
        st.session_state["fitness_profile_form_owner"] = fitness_name

    with st.form("fitness_profile_form"):
        st.markdown("**Your goal profile**")
        goal = st.selectbox(
            "Goal",
            ["lose_weight", "gain_weight"],
            format_func=lambda value: "Lose weight" if value == "lose_weight" else "Gain weight",
            key="fitness_goal",
        )
        gender = st.selectbox(
            "Gender",
            ["female", "male", "other", "prefer_not_to_say"],
            format_func=lambda value: {
                "female": "Female",
                "male": "Male",
                "other": "Other",
                "prefer_not_to_say": "Prefer not to say",
            }[value],
            key="fitness_gender",
        )
        profile_col1, profile_col2 = st.columns(2)
        with profile_col1:
            age = st.number_input(
                "Age (years)", min_value=13, max_value=100, step=1, key="fitness_age"
            )
            height_cm = st.number_input(
                "Height (cm)", min_value=100.0, max_value=250.0, step=0.5,
                key="fitness_height_cm",
            )
        with profile_col2:
            starting_weight = st.number_input(
                "Starting weight (kg)", min_value=20.0, max_value=350.0, step=0.5,
                key="fitness_start_weight",
            )
            target_weight = st.number_input(
                "Target weight (kg)", min_value=20.0, max_value=350.0, step=0.5,
                key="fitness_target_weight",
            )
        save_profile = st.form_submit_button("Save goal profile", type="primary")

    if save_profile:
        if not fitness_name:
            st.error("Enter a profile name before saving.")
        else:
            body = {
                "user_name": fitness_name,
                "goal": goal,
                "gender": gender,
                "age": int(age),
                "height_cm": float(height_cm),
                "starting_weight_kg": float(starting_weight),
                "target_weight_kg": float(target_weight),
            }
            try:
                result = progress_api("PUT", "/fitness-profile", json=body)
                st.session_state["fitness_profile"] = result["profile"]
                entries = progress_api(
                    "GET", "/progress", params={"user_name": fitness_name}
                )
                st.session_state["fitness_entries"] = entries.get("entries", [])
                st.success("Goal profile saved.")
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                st.error(str(exc))

    stored_profile = st.session_state.get("fitness_profile")
    if stored_profile and stored_profile.get("user_name") == fitness_name:
        st.divider()
        if stored_profile.get("profile_id") is not None:
            st.caption(f"Profile ID: {stored_profile['profile_id']} · Your logs use this ID.")
        goal_label = "weight loss" if stored_profile["goal"] == "lose_weight" else "weight gain"
        st.info(
            f"Saved goal: {goal_label} · starting weight {float(stored_profile['starting_weight_kg']):.1f} kg · "
            f"target {float(stored_profile['target_weight_kg']):.1f} kg."
        )
        if st.session_state.get("fitness_progress_form_owner") != fitness_name:
            previous_entries = st.session_state.get("fitness_entries", [])
            latest_weight = (
                float(previous_entries[0]["weight_kg"])
                if previous_entries
                else float(stored_profile["starting_weight_kg"])
            )
            st.session_state["fitness_entry_date"] = date.today()
            st.session_state["fitness_entry_weight"] = latest_weight
            for field in (
                "duration", "burned", "consumed", "protein", "pushups", "situps"
            ):
                st.session_state.pop(f"fitness_entry_{field}", None)
            st.session_state["fitness_progress_form_owner"] = fitness_name

        with st.form("fitness_progress_form"):
            st.markdown("**Daily check-in**")
            st.caption("One entry is stored per date. Saving again for that date replaces its values.")
            entry_date = st.date_input("Entry date", key="fitness_entry_date")
            entry_weight = st.number_input(
                "Body weight (kg)", min_value=20.0, max_value=350.0, step=0.1,
                key="fitness_entry_weight",
            )
            progress_col1, progress_col2 = st.columns(2)
            with progress_col1:
                duration_minutes = st.number_input(
                    "Workout duration (minutes, optional)", min_value=0.0, max_value=600.0,
                    value=None, step=5.0, key="fitness_entry_duration",
                )
                calories_burned = st.number_input(
                    "Calories burned (optional)", min_value=0.0, max_value=5000.0,
                    value=None, step=10.0, key="fitness_entry_burned",
                )
                calories_consumed = st.number_input(
                    "Calories consumed (optional)", min_value=0.0, max_value=15000.0,
                    value=None, step=50.0, key="fitness_entry_consumed",
                )
            with progress_col2:
                protein_grams = st.number_input(
                    "Protein consumed (grams, optional)", min_value=0.0, max_value=500.0,
                    value=None, step=5.0, key="fitness_entry_protein",
                )
                pushups = st.number_input(
                    "Push-ups completed (optional)", min_value=0, max_value=5000,
                    value=None, step=1, key="fitness_entry_pushups",
                )
                situps = st.number_input(
                    "Sit-ups completed (optional)", min_value=0, max_value=5000,
                    value=None, step=1, key="fitness_entry_situps",
                )
            save_entry = st.form_submit_button("Save daily check-in", type="primary")

        if save_entry:
            body = {
                "user_name": fitness_name,
                "entry_date": entry_date.isoformat(),
                "weight_kg": float(entry_weight),
                "workout_duration_minutes": duration_minutes,
                "calories_burned": calories_burned,
                "calories_consumed": calories_consumed,
                "protein_grams": protein_grams,
                "pushups": pushups,
                "situps": situps,
            }
            try:
                progress_api("POST", "/progress", json=body)
                entries = progress_api(
                    "GET", "/progress", params={"user_name": fitness_name}
                )
                st.session_state["fitness_entries"] = entries.get("entries", [])
                st.success("Daily check-in saved.")
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                st.error(str(exc))

        with st.expander("Food log and nutrition estimates", expanded=False):
            st.write(
                "Search the local FoodData Central catalog, enter the amount eaten, and FitPredict "
                "will estimate calories and protein from the food's values per 100 grams."
            )
            food_date = st.date_input("Food log date", value=date.today(), key="food_log_date")
            food_query = st.text_input(
                "Search foods", placeholder="For example: oatmeal or chicken", key="food_search_query"
            ).strip()
            food_matches = []
            if len(food_query) >= 2:
                try:
                    response = requests.get(
                        f"{api_url}/foods", params={"q": food_query}, timeout=20
                    )
                    result = response.json()
                    if response.ok and result.get("success"):
                        food_matches = result.get("foods", [])
                    else:
                        st.error(result.get("error", "The food catalog could not be searched."))
                except requests.RequestException as exc:
                    st.error(f"Could not reach the food catalog: {exc}")
                except ValueError:
                    st.error("The API returned a response that was not valid JSON.")
            elif food_query:
                st.caption("Enter at least two letters to search the catalog.")

            if food_matches:
                foods_by_id = {int(food["fdc_id"]): food for food in food_matches}
                selected_fdc_id = st.selectbox(
                    "Food",
                    options=list(foods_by_id),
                    format_func=lambda item_id: foods_by_id[item_id]["description"],
                    key=f"food_log_selected_fdc_id_{food_query.casefold()}",
                )
                selected_food = foods_by_id[selected_fdc_id]
                portions = selected_food.get("portions", [])
                portion_choices = [None, *portions]
                portion_index = st.selectbox(
                    "Serving size",
                    options=range(len(portion_choices)),
                    format_func=lambda index: (
                        "Enter grams"
                        if index == 0
                        else f"{portion_choices[index]['description']} "
                        f"({portion_choices[index]['grams']:g} g)"
                    ),
                    key=f"food_log_portion_{selected_fdc_id}",
                )
                default_grams = (
                    100.0
                    if portion_index == 0
                    else float(portion_choices[portion_index]["grams"])
                )
                amount_g = st.number_input(
                    "Amount eaten (grams)",
                    min_value=0.1,
                    max_value=5000.0,
                    value=default_grams,
                    step=10.0,
                    key=f"food_log_amount_{selected_fdc_id}_{portion_index}",
                )
                estimated_calories = (
                    float(selected_food["calories_kcal_per_100g"]) * amount_g / 100.0
                )
                estimated_protein = (
                    float(selected_food["protein_g_per_100g"]) * amount_g / 100.0
                )
                nutrition_cols = st.columns(2)
                nutrition_cols[0].metric("Estimated calories", f"{estimated_calories:.1f} kcal")
                nutrition_cols[1].metric("Estimated protein", f"{estimated_protein:.1f} g")
                st.caption("Catalog averages are estimates; recipes and preparation can change actual values.")
                if st.button("Add food to this day", key=f"add_food_{selected_fdc_id}"):
                    try:
                        added = progress_api(
                            "POST",
                            "/food-entries",
                            json={
                                "user_name": fitness_name,
                                "entry_date": food_date.isoformat(),
                                "fdc_id": selected_fdc_id,
                                "amount_g": float(amount_g),
                            },
                        )
                        loaded = progress_api(
                            "GET",
                            "/food-entries",
                            params={
                                "user_name": fitness_name,
                                "entry_date": food_date.isoformat(),
                            },
                        )
                        st.session_state["food_log_entries"] = loaded.get("entries", [])
                        st.session_state["food_log_totals"] = loaded.get("daily_totals", {})
                        st.session_state["food_log_owner"] = (fitness_name, food_date.isoformat())
                        estimate = added.get("estimate", {})
                        st.success(
                            f"Added {amount_g:g} g · {estimate.get('calories_kcal', 0):.1f} kcal · "
                            f"{estimate.get('protein_g', 0):.1f} g protein."
                        )
                    except (requests.RequestException, RuntimeError, ValueError) as exc:
                        st.error(str(exc))

            if st.button("Load food log for this day", key="load_food_log"):
                try:
                    loaded = progress_api(
                        "GET",
                        "/food-entries",
                        params={
                            "user_name": fitness_name,
                            "entry_date": food_date.isoformat(),
                        },
                    )
                    st.session_state["food_log_entries"] = loaded.get("entries", [])
                    st.session_state["food_log_totals"] = loaded.get("daily_totals", {})
                    st.session_state["food_log_owner"] = (fitness_name, food_date.isoformat())
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    st.error(str(exc))

            food_log_owner = st.session_state.get("food_log_owner")
            if food_log_owner == (fitness_name, food_date.isoformat()):
                totals = st.session_state.get("food_log_totals", {})
                food_entries = st.session_state.get("food_log_entries", [])
                total_cols = st.columns(2)
                total_cols[0].metric(
                    "Food log calories", f"{float(totals.get('calories_kcal', 0)):.1f} kcal"
                )
                total_cols[1].metric(
                    "Food log protein", f"{float(totals.get('protein_g', 0)):.1f} g"
                )
                if food_entries:
                    food_frame = pd.DataFrame.from_records(food_entries).rename(
                        columns={
                            "food_description": "Food",
                            "amount_g": "Amount (g)",
                            "calories_kcal": "Calories (kcal)",
                            "protein_g": "Protein (g)",
                        }
                    )
                    st.dataframe(
                        food_frame[["Food", "Amount (g)", "Calories (kcal)", "Protein (g)"]],
                        hide_index=True,
                    )
                    delete_labels = {
                        int(item["entry_id"]): (
                            f"{item['food_description']} · {float(item['amount_g']):g} g"
                        )
                        for item in food_entries
                    }
                    selected_entry_id = st.selectbox(
                        "Remove an item if needed",
                        options=list(delete_labels),
                        format_func=lambda item_id: delete_labels[item_id],
                        key=f"food_log_delete_selection_{fitness_name}_{food_date.isoformat()}",
                    )
                    if st.button("Remove selected food", key="delete_food_log_entry"):
                        try:
                            progress_api(
                                "DELETE",
                                f"/food-entries/{selected_entry_id}",
                                params={"user_name": fitness_name},
                            )
                            loaded = progress_api(
                                "GET",
                                "/food-entries",
                                params={
                                    "user_name": fitness_name,
                                    "entry_date": food_date.isoformat(),
                                },
                            )
                            st.session_state["food_log_entries"] = loaded.get("entries", [])
                            st.session_state["food_log_totals"] = loaded.get("daily_totals", {})
                            st.success("Food item removed.")
                        except (requests.RequestException, RuntimeError, ValueError) as exc:
                            st.error(str(exc))
                else:
                    st.caption("No foods have been logged for this day yet.")

        progress_rows = st.session_state.get("fitness_entries", [])
        if progress_rows:
            progress_frame = pd.DataFrame.from_records(progress_rows)
            progress_frame["entry_date"] = pd.to_datetime(
                progress_frame["entry_date"], errors="coerce"
            )
            progress_frame = progress_frame.dropna(subset=["entry_date"]).sort_values("entry_date")
            if not progress_frame.empty:
                current_weight = float(progress_frame.iloc[-1]["weight_kg"])
                weight_delta = current_weight - float(stored_profile["starting_weight_kg"])
                metric_cols = st.columns(3)
                metric_cols[0].metric("Latest weight", f"{current_weight:.1f} kg")
                metric_cols[1].metric(
                    "Distance to target",
                    f"{abs(current_weight - float(stored_profile['target_weight_kg'])):.1f} kg",
                )
                metric_cols[2].metric("Check-ins", len(progress_frame))
                st.caption(f"Weight change from starting value: {weight_delta:+.1f} kg")
                st.line_chart(
                    progress_frame.set_index("entry_date")[["weight_kg"]]
                )

                strength_frame = progress_frame.set_index("entry_date")[["pushups", "situps"]]
                if strength_frame.notna().any().any():
                    st.subheader("Strength check-ins")
                    st.line_chart(strength_frame)

                visible_columns = [
                    "entry_date", "weight_kg", "workout_duration_minutes", "calories_burned",
                    "calories_consumed", "protein_grams", "pushups", "situps",
                ]
                display_frame = (
                    progress_frame[visible_columns]
                    .sort_values("entry_date", ascending=False)
                    .rename(
                        columns={
                            "entry_date": "Date",
                            "weight_kg": "Weight (kg)",
                            "workout_duration_minutes": "Workout (min)",
                            "calories_burned": "Calories burned",
                            "calories_consumed": "Calories consumed",
                            "protein_grams": "Protein (g)",
                            "pushups": "Push-ups",
                            "situps": "Sit-ups",
                        }
                    )
                )
                st.dataframe(display_frame, hide_index=True)
                safe_name = "".join(
                    character.lower() if character.isalnum() else "-"
                    for character in fitness_name
                ).strip("-") or "profile"
                export_frame = display_frame.copy()
                for field in (
                    "goal", "gender", "age", "height_cm", "starting_weight_kg", "target_weight_kg"
                ):
                    export_frame[field] = stored_profile[field]
                st.download_button(
                    "Download progress data",
                    data=export_frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"fitpredict-progress-{safe_name}.csv",
                    mime="text/csv",
                    key="download_progress_data",
                )
        else:
            st.caption("Your saved check-ins and progress charts will appear here.")
    elif fitness_name:
        st.caption("Save or load a goal profile to begin recording daily progress.")


with analytics_tab:
    st.subheader("Your workout analytics")
    st.caption("Charts use saved workout history from the configured SQLite or MySQL database.")
    analytics_name = st.text_input("Filter by name (optional)", key="analytics_name")
    if st.button("Refresh analytics", key="refresh_analytics"):
        try:
            st.session_state["analytics_history"] = load_history(analytics_name)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.session_state["analytics_error"] = str(exc)
        else:
            st.session_state.pop("analytics_error", None)

    if st.session_state.get("analytics_error"):
        st.info(st.session_state["analytics_error"])
    analytics = st.session_state.get("analytics_history", pd.DataFrame())
    if not analytics.empty:
        analytics["created_at"] = pd.to_datetime(analytics["created_at"], errors="coerce")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Workouts", len(analytics))
        metric2.metric("Average calories burned per workout", f"{analytics['estimated_calories'].mean():.0f} kcal")
        metric3.metric("Average duration", f"{analytics['duration_minutes'].mean():.0f} min")
        daily = (
            analytics.dropna(subset=["created_at"])
            .assign(workout_date=lambda rows: rows["created_at"].dt.date)
            .groupby("workout_date", as_index=False)["estimated_calories"].sum()
        )
        if not daily.empty:
            st.plotly_chart(
                px.line(daily, x="workout_date", y="estimated_calories", markers=True,
                        title="Daily estimated calories"),
            )

with history_tab:
    st.subheader("Workout history")
    st.caption("Goal profiles, daily check-ins, food logs, and workout predictions share a profile ID. Enter a name to see one person's records.")
    history_name = st.text_input("Filter history by profile name (optional)", key="history_name").strip()
    if st.button("Load workout history", key="load_history"):
        try:
            data = progress_api("GET", "/history", params={"user_name": history_name} if history_name else None)
            st.session_state["linked_history"] = data
            st.session_state["linked_history_filter"] = history_name
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.session_state["history_error"] = str(exc)
            st.session_state.pop("linked_history", None)
        else:
            st.session_state.pop("history_error", None)

    if st.session_state.get("history_error"):
        st.info(st.session_state["history_error"])
    history = st.session_state.get("linked_history")
    if history and st.session_state.get("linked_history_filter") == history_name:
        logs = [
            ("Goal profile log", "goals", "Saved goals and changes"),
            ("Daily check-in log", "checkins", "User-entered daily check-ins"),
            ("Food log and nutrition estimates", "foods", "Logged foods with estimated calories and protein"),
            ("Workout prediction log", "workouts", "Workout calorie predictions"),
        ]
        for title, key, description in logs:
            st.subheader(title)
            records = history.get(key, [])
            if records:
                st.dataframe(pd.DataFrame.from_records(records), hide_index=True, alt=description)
            else:
                st.caption("No saved records found.")
        st.caption("Food estimates and manually entered check-in calories or protein are separate records. Unnamed workout predictions have no profile ID.")
    elif not st.session_state.get("history_error"):
        st.caption("Select Load workout history to view the saved logs.")
