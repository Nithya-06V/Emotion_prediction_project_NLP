CREATE DATABASE IF NOT EXISTS emotion_detection_db;
USE emotion_detection_db;

CREATE TABLE IF NOT EXISTS prediction_history (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    input_text TEXT NOT NULL,
    predicted_emotion VARCHAR(255) NOT NULL,
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_prediction_timestamp (predicted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;