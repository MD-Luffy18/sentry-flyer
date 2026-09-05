-- =====================================================================
-- Sentry Flyer — Onboard Relational Database Schema
-- Location: Phase 1/schema.sql
-- Purpose: Relational schema for high-speed offline storage, indexing,
--          and spatial triage of detections.
--
-- This file is the single source of truth for the schema. database.py
-- and the Phase 5 dashboard both load it at start-up; do not duplicate
-- CREATE TABLE statements elsewhere.
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
    class_name TEXT NOT NULL,      -- 'person', 'fire', 'smoke', 'floodwater', etc.
    latitude REAL NOT NULL,        -- Projected ground coordinate
    longitude REAL NOT NULL,       -- Projected ground coordinate
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    severity INTEGER NOT NULL CHECK (severity >= 0 AND severity <= 100),
    timestamp REAL NOT NULL,       -- Epoch timestamp of capture
    sensor_mode TEXT NOT NULL,     -- 'RGB', 'Thermal', or 'Fused'
    frame_filename TEXT,           -- Filename of captured frame on local storage
    cluster_id INTEGER REFERENCES fused_clusters(id) ON DELETE SET NULL
);

-- Table 3: Fused Target Clusters
-- Consolidates detections within CLUSTER_RADIUS_M (20 m) into distinct,
-- high-value triage pins. Centre/mean/max are maintained incrementally by
-- database.py so an update is O(1) regardless of how many raw detections
-- feed the cluster.
CREATE TABLE IF NOT EXISTS fused_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,
    center_latitude REAL NOT NULL,
    center_longitude REAL NOT NULL,
    mean_confidence REAL NOT NULL,
    max_severity INTEGER NOT NULL,
    detection_count INTEGER NOT NULL DEFAULT 1,
    first_seen REAL NOT NULL,
    last_updated REAL NOT NULL
);

-- =====================================================================
-- PERFORMANCE INDEXES
-- =====================================================================
-- Telemetry is queried by "latest" and by time window.
CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry_logs(timestamp);

-- Cluster matching filters by class first, then by a lat/lon bounding box.
CREATE INDEX IF NOT EXISTS idx_detections_class_spatial
    ON raw_detections(class_name, latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_detections_cluster ON raw_detections(cluster_id);
CREATE INDEX IF NOT EXISTS idx_clusters_class_spatial
    ON fused_clusters(class_name, center_latitude, center_longitude);
