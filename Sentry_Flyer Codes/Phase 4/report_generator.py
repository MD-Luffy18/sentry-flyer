"""
Sentry Flyer — Mission Reporting & SitRep Compiler
Location: Phase 4/report_generator.py
Purpose: Queries the local mission database to compile and export standardized 
         GeoJSON maps and compressed micro-bandwidth telemetry text packets.
"""

import sqlite3
import json
import os

class MissionReporter:
    def __init__(self, db_path="../Phase 1/sentry_mission.db"):
        """
        Initializes the reporter with the path to our local SQLite database.
        """
        self.db_path = db_path

    def _get_connection(self):
        """
        Establishes connection to the SQLite database.
        """
        if not os.path.exists(self.db_path):
            # Fallback path in case directory structures are absolute
            self.db_path = "sentry_mission.db"
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def export_geojson_report(self, output_path="mission_report.geojson"):
        """
        Queries fused clusters and compiles a standard spatial GeoJSON FeatureCollection.
        """
        print(f"[REPORT] Querying database to build GeoJSON SitRep: {output_path}...")
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Fetch all prioritized clusters
            cursor.execute("SELECT * FROM fused_clusters ORDER BY (max_severity * mean_confidence) DESC")
            rows = cursor.fetchall()

            # GeoJSON Schema skeleton
            geojson_data = {
                "type": "FeatureCollection",
                "features": []
            }

            for row in rows:
                # Structure properties for GIS mapping layers
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row["center_longitude"], row["center_latitude"]] # GeoJSON coordinates order is [Lon, Lat]
                    },
                    "properties": {
                        "id": row["id"],
                        "class_name": row["class_name"],
                        "confidence": round(row["mean_confidence"], 2),
                        "severity_score": row["max_severity"],
                        "detections_count": row["detection_count"],
                        "last_seen_epoch": row["last_updated"]
                    }
                }
                geojson_data["features"].append(feature)

            # Write formatted JSON output to disk
            with open(output_path, "w") as f:
                json.dump(geojson_data, f, indent=4)

            print(f"[REPORT] Successfully generated GeoJSON report containing {len(rows)} targets.")
            return True

        except sqlite3.OperationalError as e:
            print(f"[REPORT ERROR] Could not read tables. Ensure database is initialized. Details: {e}")
            return False
        finally:
            conn.close()

    def generate_micro_lora_packet(self, output_path="lora_packet.txt"):
        """
        Compresses active target clusters into an ultra-dense, comma-separated text payload 
        optimized for micro-bandwidth LoRa radio transceivers.
        Format: $SF,[Count],[ID]:[ClassID]:[Lat]:[Lon]:[Severity]-[ID]:...*Checksum
        """
        print(f"[REPORT] Compiling dense micro-bandwidth LoRa telemetry packet: {output_path}...")
        conn = self._get_connection()
        cursor = conn.cursor()

        # Simplified class mappings to compress strings into single characters
        class_mapping = {
            "person": "P",
            "fire": "F",
            "smoke": "S",
            "floodwater": "W",
            "damage": "D"
        }

        try:
            cursor.execute("SELECT id, class_name, center_latitude, center_longitude, max_severity FROM fused_clusters")
            rows = cursor.fetchall()

            if not rows:
                payload = "$SF,0,EMPTY*00"
            else:
                target_segments = []
                for row in rows:
                    class_code = class_mapping.get(row["class_name"], "U")
                    # Format float coordinates to 5 decimal places (~1.1 meter accuracy) to save valuable characters
                    lat_str = f"{row['center_latitude']:.5f}"
                    lon_str = f"{row['center_longitude']:.5f}"
                    
                    segment = f"{row['id']}:{class_code}:{lat_str}:{lon_str}:{row['max_severity']}"
                    target_segments.append(segment)

                # Assemble the packet body
                payload_body = f"SF,{len(rows)}," + "-".join(target_segments)
                
                # Compute standard NMEA-style XOR checksum of payload body characters
                checksum = 0
                for char in payload_body:
                    checksum ^= ord(char)
                
                # Append sentence structure markers
                payload = f"${payload_body}*{checksum:02X}"

            # Save the compressed text payload
            with open(output_path, "w") as f:
                f.write(payload)

            print(f"[REPORT] Successfully packed {len(rows)} targets into {len(payload)} bytes of LoRa payload.")
            return payload

        except sqlite3.OperationalError as e:
            print(f"[REPORT ERROR] Database read failed. Details: {e}")
            return None
        finally:
            conn.close()


# =====================================================================
# SYSTEM INTEGRATION TEST
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Initializing Sentry Flyer Mission Reporter...")
    
    # Check if a temporary database needs to be built for testing
    db_file = "sentry_mission_temp.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Setup mock tables matching our schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fused_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            center_latitude REAL NOT NULL,
            center_longitude REAL NOT NULL,
            mean_confidence REAL NOT NULL,
            max_severity INTEGER NOT NULL,
            detection_count INTEGER DEFAULT 1,
            last_updated REAL NOT NULL
        )
    """)
    conn.commit()
    
    # Inject 2 mock survivor detections
    cursor.execute("DELETE FROM fused_clusters")
    cursor.execute("""
        INSERT INTO fused_clusters (class_name, center_latitude, center_longitude, mean_confidence, max_severity, detection_count, last_updated)
        VALUES ('person', 25.31025, 78.48810, 0.94, 98, 4, 1711111111.0)
    """)
    cursor.execute("""
        INSERT INTO fused_clusters (class_name, center_latitude, center_longitude, mean_confidence, max_severity, detection_count, last_updated)
        VALUES ('fire', 25.31140, 78.48930, 0.89, 85, 2, 1711111115.0)
    """)
    conn.commit()
    conn.close()

    # Instantiate reporter pointing to our mock database
    reporter = MissionReporter(db_path=db_file)
    
    # Run exports
    reporter.export_geojson_report(output_path="test_mission_report.geojson")
    lora_data = reporter.generate_micro_lora_packet(output_path="test_lora_packet.txt")
    
    print("\n=== GENERATED LORA PACKET PAYLOAD ===")
    print(lora_data)
    print(f"Total String Size: {len(lora_data)} characters/bytes")
    print("======================================\n")
    
    # Cleanup temporary test database files
    try:
        os.remove(db_file)
        os.remove("test_mission_report.geojson")
        os.remove("test_lora_packet.txt")
    except OSError:
        pass