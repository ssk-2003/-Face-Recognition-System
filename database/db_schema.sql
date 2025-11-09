-- FRS Database Schema
-- Generated: 2025-11-09 19:13:34

CREATE TABLE IF NOT EXISTS identity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) UNIQUE NOT NULL,
    image_path VARCHAR(512) NOT NULL,
    embedding_json TEXT NOT NULL,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    num_faces_detected INTEGER,
    recognized_identity VARCHAR(255),
    confidence FLOAT,
    processing_time_ms FLOAT
);

CREATE INDEX idx_identity_name ON identity(name);
CREATE INDEX idx_detection_timestamp ON detection_log(timestamp);
