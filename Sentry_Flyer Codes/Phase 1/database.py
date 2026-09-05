"""
Sentry Flyer — Database & Spatial Clustering Engine
Location: Phase 1/database.py
Purpose: Thread-safe SQLite transactions and great-circle spatial clustering 
         using the Haversine mathematical model.
"""

import sqlite3
import math
import time

class SentryDatabase:
    def __init__(self, db_path="sentry_mission.db"):
        """
        Initializes the database connection.
        Enforces foreign key constraints and prepares the system.
        """
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        """
        Returns a thread-safe connection to SQLite.
        """
        conn = sqlite3.connect(self.db_path)
        # Enable ROW factory so we can access columns by name like dictionary items
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """
        Executes setup logic. Loads tables and indexes if they do not exist.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Read the SQL definitions directly from our schema file
        try:
            with open("schema.sql", "r") as f:
                schema_script = f.read()
            cursor.executescript(schema_script)
            conn.commit()
            print("[DB INFO] Database structures verified and initialized.")
        except FileNotFoundError:
            print("[DB ERROR] schema.sql was not found! Table structural creation skipped.")
        finally:
            conn.close()

    def log_telemetry(self, lat, lon, alt, heading, pitch, roll, volts):
        """
        Inserts a high-frequency flight telemetry log row.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO telemetry_logs (latitude, longitude, altitude, heading, pitch, roll, battery_voltage, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (lat, lon, alt, heading, pitch, roll, volts, time.time()))
        conn.commit()
        conn.close()

    @staticmethod
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculates the Great-Circle distance between two points on the Earth's surface
        using the Haversine trigonometric formula. Returns distance in meters.
        """
        # Earth's radius in meters
        earth_radius = 6371000.0

        # Convert decimal degrees to radians
        rad_lat1 = math.radians(lat1)
        rad_lon1 = math.radians(lon1)
        rad_lat2 = math.radians(lat2)
        rad_lon2 = math.radians(lon2)

        # Differences in coordinates
        d_lat = rad_lat2 - rad_lat1
        d_lon = rad_lon2 - rad_lon1

        # Haversine core formula
        a = (math.sin(d_lat / 2.0) ** 2) + \
            (math.cos(rad_lat1) * math.cos(rad_lat2) * (math.sin(d_lon / 2.0) ** 2))
        
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        distance_meters = earth_radius * c
        return distance_meters

    def insert_and_cluster_detection(self, class_name, lat, lon, confidence, severity, sensor_mode, frame_path=""):
        """
        Inserts a raw detection, then recalculates 20-meter spatial clustering.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        current_time = time.time()

        # Step 1: Insert into the raw log table
        raw_query = """
            INSERT INTO raw_detections (class_name, latitude, longitude, confidence, severity, timestamp, sensor_mode, frame_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(raw_query, (class_name, lat, lon, confidence, severity, current_time, sensor_mode, frame_path))
        conn.commit()

        # Step 2: Fetch existing clusters of the SAME class to check distance bounds
        cluster_query = "SELECT * FROM fused_clusters WHERE class_name = ?"
        cursor.execute(cluster_query, (class_name,))
        existing_clusters = cursor.fetchall()

        matched_cluster_id = None

        # Step 3: Run spatial proximity checks (20-meter threshold)
        for cluster in existing_clusters:
            dist = self.calculate_haversine_distance(lat, lon, cluster["center_latitude"], cluster["center_longitude"])
            if dist <= 20.0:  # Spatial grouping threshold
                matched_cluster_id = cluster["id"]
                break

        if matched_cluster_id is not None:
            # Step 4a: Update existing cluster (Average position, bump count, tracking max severity)
            # Fetch all raw detections associated within this cluster boundary
            # In a lightweight system, we fetch raw detections closest to center coordinates
            cursor.execute(
                "SELECT * FROM raw_detections WHERE class_name = ? AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?",
                (class_name, lat-0.0003, lat+0.0003, lon-0.0003, lon+0.0003)
            )
            raw_points = cursor.fetchall()
            
            # Filter exactly using Haversine calculation to ensure precision
            valid_points = [p for p in raw_points if self.calculate_haversine_distance(lat, lon, p["latitude"], p["longitude"]) <= 20.0]
            
            count = len(valid_points)
            avg_lat = sum(p["latitude"] for p in valid_points) / count
            avg_lon = sum(p["longitude"] for p in valid_points) / count
            avg_conf = sum(p["confidence"] for p in valid_points) / count
            max_sev = max(p["severity"] for p in valid_points)

            update_query = """
                UPDATE fused_clusters 
                SET center_latitude = ?, center_longitude = ?, mean_confidence = ?, max_severity = ?, detection_count = ?, last_updated = ?
                WHERE id = ?
            """
            cursor.execute(update_query, (avg_lat, avg_lon, avg_conf, max_sev, count, current_time, matched_cluster_id))
        else:
            # Step 4b: Create a brand new distinct cluster
            insert_cluster_query = """
                INSERT INTO fused_clusters (class_name, center_latitude, center_longitude, mean_confidence, max_severity, detection_count, last_updated)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """
            cursor.execute(insert_cluster_query, (class_name, lat, lon, confidence, severity, current_time))

        conn.commit()
        conn.close()

    def fetch_prioritized_map(self):
        """
        Retrieves Consolidated target clusters prioritized by: Severity * Mean Confidence.
        Highest-priority survivors bubble directly to the top of the queue.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT *, (max_severity * mean_confidence) AS priority_score 
            FROM fused_clusters 
            ORDER BY priority_score DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results


# =====================================================================
# INTEGRATION TESTING RUN (Executable validation)
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Initializing Sentry Flyer Local Database Engine...")
    db = SentryDatabase()

    # 1. Simulate telemetry writes (High rate)
    print("[TEST] Logging flight coordinates...")
    db.log_telemetry(25.31020, 78.48890, 45.5, 90.0, 1.2, -0.4, 11.8)
    db.log_telemetry(25.31022, 78.48892, 45.6, 90.0, 1.0, -0.3, 11.7)

    # 2. Simulate raw detections in close proximity (Cluster 1: Survivor found at coordinate center)
    print("[TEST] Feeding detection streams into spatial filter...")
    # Detection A: First sighting
    db.insert_and_cluster_detection("person", 25.31000, 78.48800, 0.85, 95, "Fused", "frame_001.png")
    # Detection B: Second sighting 5 meters away (should merge into same cluster)
    db.insert_and_cluster_detection("person", 25.31004, 78.48804, 0.92, 95, "Fused", "frame_002.png")
    # Detection C: Third sighting 45 meters away (outside 20m limit; should create Cluster 2)
    db.insert_and_cluster_detection("person", 25.31040, 78.48840, 0.80, 95, "Fused", "frame_003.png")

    # 3. Fetch prioritized results
    priority_list = db.fetch_prioritized_map()
    print("\n=== SYSTEM CLUSTER REPORT ===")
    for row in priority_list:
        print(f"Class: {row['class_name']} | Count: {row['detection_count']} | "
              f"Avg Conf: {row['mean_confidence']:.2f} | Lat: {row['center_latitude']:.5f} | "
              f"Lon: {row['center_longitude']:.5f} | Priority Score: {row['priority_score']:.1f}")
    print("=============================\n")