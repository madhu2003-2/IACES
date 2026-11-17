CREATE DATABASE IF NOT EXISTS fitpredict_ai
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE fitpredict_ai;

CREATE TABLE IF NOT EXISTS users (
  user_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  display_name VARCHAR(100) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id),
  UNIQUE KEY uq_users_display_name (display_name)
);

CREATE TABLE IF NOT EXISTS workout_sessions (
  workout_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id BIGINT UNSIGNED NULL,
  age TINYINT UNSIGNED NOT NULL,
  gender ENUM('male', 'female') NOT NULL,
  weight_kg DECIMAL(6,2) NOT NULL,
  duration_minutes DECIMAL(7,2) NOT NULL,
  heart_rate_bpm SMALLINT UNSIGNED NOT NULL,
  estimated_calories DECIMAL(9,2) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (workout_id),
  KEY idx_workout_sessions_created_at (created_at),
  KEY idx_workout_sessions_user_created (user_id, created_at),
  CONSTRAINT fk_workout_sessions_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE SET NULL,
  CONSTRAINT chk_workout_age CHECK (age BETWEEN 10 AND 100),
  CONSTRAINT chk_workout_weight CHECK (weight_kg BETWEEN 20 AND 350),
  CONSTRAINT chk_workout_duration CHECK (duration_minutes BETWEEN 1 AND 300),
  CONSTRAINT chk_workout_heart_rate CHECK (heart_rate_bpm BETWEEN 30 AND 240),
  CONSTRAINT chk_workout_calories CHECK (estimated_calories >= 0)
);

CREATE TABLE IF NOT EXISTS fitness_profiles (
  user_id BIGINT UNSIGNED NOT NULL,
  goal ENUM('lose_weight', 'gain_weight') NOT NULL,
  gender ENUM('female', 'male', 'other', 'prefer_not_to_say') NOT NULL,
  age TINYINT UNSIGNED NOT NULL,
  height_cm DECIMAL(5,2) NOT NULL,
  starting_weight_kg DECIMAL(6,2) NOT NULL,
  target_weight_kg DECIMAL(6,2) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id),
  CONSTRAINT fk_fitness_profiles_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE CASCADE,
  CONSTRAINT chk_profile_age CHECK (age BETWEEN 13 AND 100),
  CONSTRAINT chk_profile_height CHECK (height_cm BETWEEN 100 AND 250),
  CONSTRAINT chk_profile_start_weight CHECK (starting_weight_kg BETWEEN 20 AND 350),
  CONSTRAINT chk_profile_target_weight CHECK (target_weight_kg BETWEEN 20 AND 350)
);

CREATE TABLE IF NOT EXISTS goal_profile_log (
  goal_log_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  user_id BIGINT UNSIGNED NOT NULL,
  legacy_user_id BIGINT UNSIGNED NULL UNIQUE,
  goal VARCHAR(20) NOT NULL,
  gender VARCHAR(20) NOT NULL,
  age SMALLINT NOT NULL,
  height_cm DECIMAL(6,2) NOT NULL,
  starting_weight_kg DECIMAL(6,2) NOT NULL,
  target_weight_kg DECIMAL(6,2) NOT NULL,
  saved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_goal_log_user (user_id, saved_at),
  CONSTRAINT fk_goal_log_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS progress_entries (
  entry_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id BIGINT UNSIGNED NOT NULL,
  entry_date DATE NOT NULL,
  weight_kg DECIMAL(6,2) NOT NULL,
  workout_duration_minutes DECIMAL(7,2) NULL,
  calories_burned DECIMAL(9,2) NULL,
  calories_consumed DECIMAL(9,2) NULL,
  protein_grams DECIMAL(7,2) NULL,
  pushups SMALLINT UNSIGNED NULL,
  situps SMALLINT UNSIGNED NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (entry_id),
  UNIQUE KEY uq_progress_user_date (user_id, entry_date),
  CONSTRAINT fk_progress_entries_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE CASCADE,
  CONSTRAINT chk_progress_weight CHECK (weight_kg BETWEEN 20 AND 350),
  CONSTRAINT chk_progress_duration CHECK (workout_duration_minutes IS NULL OR workout_duration_minutes BETWEEN 0 AND 600),
  CONSTRAINT chk_progress_burned CHECK (calories_burned IS NULL OR calories_burned BETWEEN 0 AND 5000),
  CONSTRAINT chk_progress_consumed CHECK (calories_consumed IS NULL OR calories_consumed BETWEEN 0 AND 15000),
  CONSTRAINT chk_progress_protein CHECK (protein_grams IS NULL OR protein_grams BETWEEN 0 AND 500),
  CONSTRAINT chk_progress_pushups CHECK (pushups IS NULL OR pushups BETWEEN 0 AND 5000),
  CONSTRAINT chk_progress_situps CHECK (situps IS NULL OR situps BETWEEN 0 AND 5000)
);

CREATE TABLE IF NOT EXISTS food_entries (
  entry_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id BIGINT UNSIGNED NOT NULL,
  entry_date DATE NOT NULL,
  fdc_id BIGINT UNSIGNED NOT NULL,
  food_description VARCHAR(500) NOT NULL,
  amount_g DECIMAL(8,2) NOT NULL,
  calories_kcal DECIMAL(9,2) NOT NULL,
  protein_g DECIMAL(8,2) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (entry_id),
  KEY idx_food_entries_user_date (user_id, entry_date),
  CONSTRAINT fk_food_entries_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE CASCADE,
  CONSTRAINT chk_food_amount CHECK (amount_g BETWEEN 0.1 AND 5000),
  CONSTRAINT chk_food_calories CHECK (calories_kcal >= 0),
  CONSTRAINT chk_food_protein CHECK (protein_g >= 0)
);
