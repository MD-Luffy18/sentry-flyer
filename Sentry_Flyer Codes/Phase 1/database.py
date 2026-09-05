"""
Sentry Flyer — Database & Spatial Clustering Engine
Location: Phase 1/database.py
Purpose: SQLite persistence for telemetry and detections, plus incremental
         great-circle (Haversine) clustering of detections into triage pins.

Design notes
------------
* One short-lived connection per operation. SQLite handles this cheaply and it
  keeps the class safe to call from any thread.
* WAL journal mode so the Phase 5 dashboard can read while the drone writes.
* Cluster statistics (centre, mean confidence, max severity) are maintained
  incrementally, so inserting a detection is O(clusters-of-that-class in a
  ~20 m box), not O(all raw detections nearby).
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("sentry.db")

EARTH_RADIUS_M = 6_371_000.0
METERS_PER_DEG_LAT = 111_111.0

# Detections of the same class within this distance are merged into one pin.
CLUSTER_RADIUS_M = 20.0

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    d_lat = rlat2 - rlat1
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(d_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def degree_box(lat: float, radius_m: float) -> tuple[float, float]:
    """
    Half-widths (d_lat, d_lon) in degrees of a box that fully contains a circle
    of `radius_m` metres centred at latitude `lat`. Used as a cheap indexed
    pre-filter before the exact Haversine test.
    """
    d_lat = radius_m / METERS_PER_DEG_LAT
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)  # avoid blow-up at the poles
    d_lon = radius_m / (METERS_PER_DEG_LAT * cos_lat)
    return d_lat, d_lon


class SentryDatabase:
    def __init__(self, db_path: str | Path = "sentry_mission.db", schema_path: str | Path = SCHEMA_PATH):
        self.db_path = str(db_path)
        self.schema_path = Path(schema_path)
        self._init_db()

    # ------------------------------------------------------------------ #
    # Connection handling
    # ------------------------------------------------------------------ #
    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and rolls back on error."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indexes from schema.sql if they do not exist."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"schema.sql not found at {self.schema_path}")
        schema_script = self.schema_path.read_text(encoding="utf-8")
        with self._connection() as conn:
            # WAL lets the ground-station reader coexist with the onboard writer.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(schema_script)
        log.info("Database schema verified at %s", self.db_path)

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    def log_telemetry(self, lat, lon, alt, heading, pitch, roll, volts, timestamp: Optional[float] = None) -> None:
        """Insert one flight telemetry row."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO telemetry_logs
                    (latitude, longitude, altitude, heading, pitch, roll, battery_voltage, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (lat, lon, alt, heading, pitch, roll, volts, timestamp or time.time()),
            )

    def latest_telemetry(self) -> Optional[sqlite3.Row]:
        with self._connection() as conn:
            return conn.execute("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT 1").fetchone()

    # ------------------------------------------------------------------ #
    # Detections and clustering
    # ------------------------------------------------------------------ #
    @staticmethod
    def calculate_haversine_distance(lat1, lon1, lat2, lon2) -> float:
        """Backwards-compatible alias for :func:`haversine_m`."""
        return haversine_m(lat1, lon1, lat2, lon2)

    def insert_and_cluster_detection(
        self,
        class_name: str,
        lat: float,
        lon: float,
        confidence: float,
        severity: int,
        sensor_mode: str,
        frame_path: str = "",
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Insert a raw detection and merge it into the nearest same-class cluster
        within CLUSTER_RADIUS_M, or open a new cluster. Returns the cluster id.

        Everything happens in a single transaction so a crash mid-way cannot
        leave a raw detection without a cluster.
        """
        now = timestamp or time.time()
        d_lat, d_lon = degree_box(lat, CLUSTER_RADIUS_M)

        with self._connection() as conn:
            # Indexed bounding-box pre-filter, then exact distance, nearest wins.
            candidates = conn.execute(
                """
                SELECT id, center_latitude, center_longitude, mean_confidence, max_severity, detection_count
                FROM fused_clusters
                WHERE class_name = ?
                  AND center_latitude  BETWEEN ? AND ?
                  AND center_longitude BETWEEN ? AND ?
                """,
                (class_name, lat - d_lat, lat + d_lat, lon - d_lon, lon + d_lon),
            ).fetchall()

            best: Optional[sqlite3.Row] = None
            best_dist = CLUSTER_RADIUS_M
            for c in candidates:
                dist = haversine_m(lat, lon, c["center_latitude"], c["center_longitude"])
                if dist <= best_dist:
                    best, best_dist = c, dist

            if best is not None:
                n = best["detection_count"]
                new_n = n + 1
                # Incremental running means keep the update O(1).
                new_lat = (best["center_latitude"] * n + lat) / new_n
                new_lon = (best["center_longitude"] * n + lon) / new_n
                new_conf = (best["mean_confidence"] * n + confidence) / new_n
                new_sev = max(best["max_severity"], severity)
                conn.execute(
                    """
                    UPDATE fused_clusters
                    SET center_latitude = ?, center_longitude = ?, mean_confidence = ?,
                        max_severity = ?, detection_count = ?, last_updated = ?
                    WHERE id = ?
                    """,
                    (new_lat, new_lon, new_conf, new_sev, new_n, now, best["id"]),
                )
                cluster_id = best["id"]
            else:
                cur = conn.execute(
                    """
                    INSERT INTO fused_clusters
                        (class_name, center_latitude, center_longitude, mean_confidence,
                         max_severity, detection_count, first_seen, last_updated)
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (class_name, lat, lon, confidence, severity, now, now),
                )
                cluster_id = cur.lastrowid

            conn.execute(
                """
                INSERT INTO raw_detections
                    (class_name, latitude, longitude, confidence, severity, timestamp,
                     sensor_mode, frame_filename, cluster_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (class_name, lat, lon, confidence, severity, now, sensor_mode, frame_path, cluster_id),
            )

        return int(cluster_id)

    def fetch_prioritized_map(self) -> list[sqlite3.Row]:
        """
        Clusters ordered by priority = max_severity * mean_confidence, so a
        confidently-seen survivor outranks a hazard-only hit.
        """
        with self._connection() as conn:
            return conn.execute(
                """
                SELECT *, (max_severity * mean_confidence) AS priority_score
                FROM fused_clusters
                ORDER BY priority_score DESC, last_updated DESC
                """
            ).fetchall()


# =====================================================================
# INTEGRATION TESTING RUN (Executable validation)
# =====================================================================
if __name__ == "__main__":
    import os
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("[TEST] Initializing Sentry Flyer Local Database Engine...")
    tmp_db = os.path.join(tempfile.mkdtemp(), "sentry_mission.db")
    db = SentryDatabase(tmp_db)

    print("[TEST] Logging flight coordinates...")
    db.log_telemetry(25.31020, 78.48890, 45.5, 90.0, 1.2, -0.4, 11.8)
    db.log_telemetry(25.31022, 78.48892, 45.6, 90.0, 1.0, -0.3, 11.7)

    print("[TEST] Feeding detection streams into spatial filter...")
    # A: first sighting. B: 5 m away (merges). C: ~60 m away (new cluster).
    db.insert_and_cluster_detection("person", 25.31000, 78.48800, 0.85, 95, "Fused", "frame_001.png")
    db.insert_and_cluster_detection("person", 25.31004, 78.48804, 0.92, 95, "Fused", "frame_002.png")
    db.insert_and_cluster_detection("person", 25.31040, 78.48840, 0.80, 95, "Fused", "frame_003.png")

    print("\n=== SYSTEM CLUSTER REPORT ===")
    for row in db.fetch_prioritized_map():
        print(
            f"Class: {row['class_name']} | Count: {row['detection_count']} | "
            f"Avg Conf: {row['mean_confidence']:.2f} | Lat: {row['center_latitude']:.5f} | "
            f"Lon: {row['center_longitude']:.5f} | Priority Score: {row['priority_score']:.1f}"
        )
    print("=============================\n")
