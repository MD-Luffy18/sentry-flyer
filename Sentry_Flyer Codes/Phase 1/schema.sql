-- =====================================================================
-- Sentry Flyer — Onboard Relational Database Schema
-- Location: Phase 1/schema.sql
-- Purpose: Optimized relational schema for high-speed offline storage, 
--          indexing, and spatial coordinates triage.
-- =====================================================================

-- Table 1: Raw Drone Telemetry Logs
-- Logs the real-time flight path and orientation of Sentry Flyer.
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    altitude REAL NOT NULL,        -- Height in meters from barometer
    heading REAL NOT NULL,         -- Yaw angle from compass (0 to 359 degrees)
    pitch REAL NOT NULL,           -- Pitch angle from IMU
    roll REAL NOT NULL,            -- Roll angle from IMU
    battery_voltage REAL NOT NULL, -- Flight battery voltage monitor
    timestamp REAL NOT NULL        -- Epoch time (seconds since 1970-01-01)
);

-- Table 2: Raw Detections Log
-- Records every single frame prediction from the onboard Edge AI models.
CREATE TABLE IF NOT EXISTS raw_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,      -- 'person', 'fire', 'smoke', 'flood', etc.
    latitude REAL NOT NULL,        -- Projected ground coordinate
    longitude REAL NOT NULL,       -- Projected ground coordinate
    confidence REAL NOT NULL,      -- Model confidence output (0.00 to 1.00)
    severity INTEGER NOT NULL,     -- Responder urgency score (0 to 100)
    timestamp REAL NOT NULL,       -- Epoch timestamp of capture
    sensor_mode TEXT NOT NULL,     -- 'RGB', 'Thermal', or 'Fused'
    frame_filename TEXT            -- Filename of captured frame on local storage
);

-- Table 3: Fused Target Clusters
-- Consolidates spatial coordinates within 20m into distinct, high-value triage pins.
CREATE TABLE IF NOT EXISTS fused_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,
    center_latitude REAL NOT NULL,
    center_longitude REAL NOT NULL,
    mean_confidence REAL NOT NULL,
    max_severity INTEGER NOT NULL,
    detection_count INTEGER DEFAULT 1,
    last_updated REAL NOT NULL
);

-- =====================================================================
-- DATABASE PERFORMANCE INDEXES (High-Speed Spatial & Temporal Queries)
-- =====================================================================
-- These indexes prevent database slowdowns during high-frequency telemetry tracking 
-- and allow instant bounding-box queries for map rendering.
CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_spatial ON raw_detections(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_detections_class ON raw_detections(class_name);
CREATE INDEX IF NOT EXISTS idx_clusters_spatial ON fused_clusters(center_latitude, center_longitude);