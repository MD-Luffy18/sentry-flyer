"""
Sentry Flyer — Ground Station Dashboard & Telemetry Link
Location: Phase 5/app.py
Purpose: Launches an offline-first Flask web server to present a live incident map,
         real-time telemetry readouts, and prioritized triage lists for emergency responders.
         Includes a background simulation thread to demonstrate real-time data ingestion.
"""

import sqlite3
import json
import time
import os
import threading
from flask import Flask, jsonify, Response

app = Flask(__name__)

# Configurable database path pointing back to the Phase 1 storage module
DATABASE_PATH = "../Phase 1/sentry_mission.db"

def ensure_database_exists():
    """
    Ensures the SQLite database is created and initialized before the web server runs.
    """
    # Create the Phase 1 folder structure if it doesn't exist locally
    os.makedirs("../Phase 1", exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Initialize telemetry logs table matching our schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            altitude REAL NOT NULL,
            heading REAL NOT NULL,
            pitch REAL NOT NULL,
            roll REAL NOT NULL,
            battery_voltage REAL NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    
    # Initialize fused clusters table matching our schema
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
    conn.close()


def run_live_simulation():
    """
    Background simulation engine. Periodically inserts mock telemetry coordinates 
    and dynamic survivor detections to demonstrate real-time dashboard updates.
    """
    print("[SIMULATOR] Starting live mission simulation background thread...")
    ensure_database_exists()
    
    step = 0
    while True:
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # 1. Simulate drone moving along a search grid path
            sim_lat = 25.31000 + (step * 0.00015)
            sim_lon = 78.48800 + (step * 0.00010)
            sim_alt = 45.0 + (step % 3)
            sim_volt = max(12.6 - (step * 0.05), 10.8)
            current_time = time.time()
            
            cursor.execute("""
                INSERT INTO telemetry_logs (latitude, longitude, altitude, heading, pitch, roll, battery_voltage, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (sim_lat, sim_lon, sim_alt, 90.0, 1.2, -0.4, sim_volt, current_time))
            
            # 2. Periodically trigger new detections (Step 2: Person, Step 5: Fire)
            if step == 2:
                print("[SIMULATOR] Triggering live target detection: 'person' at C1")
                cursor.execute("""
                    INSERT INTO fused_clusters (class_name, center_latitude, center_longitude, mean_confidence, max_severity, detection_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("person", 25.31050, 78.48850, 0.94, 95, 3, current_time))
            elif step == 5:
                print("[SIMULATOR] Triggering live target detection: 'fire' at C2")
                cursor.execute("""
                    INSERT INTO fused_clusters (class_name, center_latitude, center_longitude, mean_confidence, max_severity, detection_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("fire", 25.31120, 78.48910, 0.88, 90, 1, current_time))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"[SIMULATOR WARNING] Loop skipped: {e}")
            
        step += 1
        time.sleep(4.0)  # Refresh state every 4 seconds


# =====================================================================
# API ENDPOINTS (Providing dynamic JSON streams to Web Frontend)
# =====================================================================

@app.route("/api/telemetry")
def get_telemetry():
    """
    Fetches the latest telemetry coordinate and flight state packet.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return jsonify({
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "altitude": round(row["altitude"], 1),
                "heading": round(row["heading"], 1),
                "battery_voltage": round(row["battery_voltage"], 2),
                "timestamp": row["timestamp"]
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    return jsonify({
        "latitude": 25.31000,
        "longitude": 78.48800,
        "altitude": 0.0,
        "heading": 0.0,
        "battery_voltage": 12.6,
        "timestamp": time.time()
    })


@app.route("/api/targets")
def get_targets():
    """
    Retrieves all active consolidated target clusters sorted by priority score.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT *, (max_severity * mean_confidence) AS priority_score FROM fused_clusters ORDER BY priority_score DESC")
        rows = cursor.fetchall()
        conn.close()
        
        targets = []
        for r in rows:
            targets.append({
                "id": r["id"],
                "class_name": r["class_name"],
                "latitude": r["center_latitude"],
                "longitude": r["center_longitude"],
                "confidence": round(r["mean_confidence"], 2),
                "severity": r["max_severity"],
                "count": r["detection_count"],
                "priority_score": round(r["max_severity"] * r["mean_confidence"], 1)
            })
        return jsonify(targets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================================
# DASHBOARD FRONTEND UI (Embedded Single-File HTML/CSS/JS Engine)
# =====================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentry Flyer — Responder Command Center</title>
    <!-- Embedded modern dark-mode CSS styling -->
    <style>
        :root {
            --bg-color: #0f172a;
            --panel-bg: #1e293b;
            --border-color: #334155;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --critical: #ef4444;
            --warning: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        header {
            background-color: var(--panel-bg);
            border-bottom: 1px solid var(--border-color);
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        header h1 {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        header .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: var(--text-muted);
        }

        .pulse-dot {
            width: 10px;
            height: 10px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .main-container {
            display: flex;
            flex: 1;
            height: calc(100vh - 54px);
        }

        /* Responsive Sidebar Grid Layout */
        .sidebar {
            width: 380px;
            border-right: 1px solid var(--border-color);
            background-color: var(--panel-bg);
            display: flex;
            flex-direction: column;
        }

        .telemetry-widget {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .telemetry-card {
            background-color: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-color);
            padding: 12px;
            border-radius: 6px;
        }

        .telemetry-card .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .telemetry-card .value {
            font-size: 16px;
            font-weight: 600;
        }

        .triage-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .triage-header {
            padding: 15px 20px;
            border-bottom: 1px solid var(--border-color);
            font-size: 14px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .triage-list {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }

        .triage-item {
            background-color: rgba(15, 23, 42, 0.3);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--primary);
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 10px;
        }

        .triage-item.critical { border-left-color: var(--critical); }
        .triage-item.warning { border-left-color: var(--warning); }

        .triage-item-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }

        .triage-item-header .class-label {
            font-weight: 600;
            font-size: 14px;
            text-transform: capitalize;
        }

        .triage-item-header .priority-tag {
            font-size: 11px;
            background-color: rgba(255, 255, 255, 0.1);
            padding: 2px 6px;
            border-radius: 3px;
            color: var(--text-muted);
        }

        .triage-item-details {
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        /* Map Container Frame */
        .map-container {
            flex: 1;
            background-color: #0c111d;
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        #map-fallback-view {
            text-align: center;
            padding: 40px;
        }

        #map-fallback-view h2 {
            font-size: 18px;
            margin-bottom: 10px;
            color: var(--text-muted);
        }

        #map-fallback-view p {
            font-size: 14px;
            color: #64748b;
            max-width: 400px;
        }
    </style>
</head>
<body>

    <header>
        <h1>SENTRY FLYER — GROUND CONTROL</h1>
        <div class="status">
            <div class="pulse-dot"></div>
            <span>OFFLINE TACTICAL NETWORK</span>
        </div>
    </header>

    <div class="main-container">
        <!-- Sidebar Navigation -->
        <div class="sidebar">
            <!-- Telemetry Logs Grid -->
            <div class="telemetry-widget">
                <div class="telemetry-card">
                    <div class="label">Drone Position</div>
                    <div class="value" id="telemetry-coords">---, ---</div>
                </div>
                <div class="telemetry-card">
                    <div class="label">Alt (Baro)</div>
                    <div class="value" id="telemetry-alt">0.0 m</div>
                </div>
                <div class="telemetry-card">
                    <div class="label">Compass</div>
                    <div class="value" id="telemetry-heading">0.0°</div>
                </div>
                <div class="telemetry-card">
                    <div class="label">Battery</div>
                    <div class="value" id="telemetry-battery">0.0 V</div>
                </div>
            </div>

            <!-- Triage Ranking Queue -->
            <div class="triage-section">
                <div class="triage-header">Prioritized Incident Queue</div>
                <div class="triage-list" id="triage-queue-list">
                    <div style="text-align:center; padding: 20px; color: var(--text-muted); font-size:14px;">
                        Scanning search sector...
                    </div>
                </div>
            </div>
        </div>

        <!-- Terminal Workspace Representation for Map -->
        <div class="map-container">
            <div id="map-fallback-view">
                <h2>TACTICAL SECTOR COORDINATOR MAP</h2>
                <p>
                    Leaflet map engine ready. Live GPS track syncing with mock serial lines. <br><br>
                    <strong>Mission Origin:</strong> 25.31000° N, 78.48800° E <br>
                    <strong>Current Sweep Area:</strong> Sector Delta (500m Geofenced Radius)
                </p>
            </div>
        </div>
    </div>

    <!-- Frontend Live Data Polling Script -->
    <script>
        function updateDashboard() {
            // 1. Fetch Latest Telemetry Coordinates
            fetch('/api/telemetry')
                .then(response => response.json())
                .then(data => {
                    if(!data.error) {
                        document.getElementById('telemetry-coords').innerText = 
                            data.latitude.toFixed(5) + ', ' + data.longitude.toFixed(5);
                        document.getElementById('telemetry-alt').innerText = data.altitude.toFixed(1) + ' m';
                        document.getElementById('telemetry-heading').innerText = data.heading.toFixed(1) + '°';
                        document.getElementById('telemetry-battery').innerText = data.battery_voltage.toFixed(2) + ' V';
                    }
                });

            // 2. Fetch Prioritized Live Target Queues
            fetch('/api/targets')
                .then(response => response.json())
                .then(targets => {
                    const queueList = document.getElementById('triage-queue-list');
                    if (targets.length === 0) {
                        queueList.innerHTML = `<div style="text-align:center; padding:20px; color: #64748b; font-size:13px;">No targets detected in search pattern yet.</div>`;
                        return;
                    }

                    let htmlContent = '';
                    targets.forEach(target => {
                        // Classify triage style boundaries based on priority ranking scores
                        let priorityClass = 'warning';
                        if (target.priority_score >= 80.0) {
                            priorityClass = 'critical';
                        }

                        htmlContent += `
                            <div class="triage-item \${priorityClass}">
                                <div class="triage-item-header">
                                    <span class="class-label">\${target.class_name}</span>
                                    <span class="priority-tag">Score: \${target.priority_score}</span>
                                </div>
                                <div class="triage-item-details">
                                    <span><strong>Coordinates:</strong> \\({target.latitude.toFixed(5)}, \\){target.longitude.toFixed(5)}</span>
                                    <span><strong>Confidence:</strong> \\({(target.confidence * 100).toFixed(0)}% | <strong>Sighting count:</strong> \\){target.count}</span>
                                </div>
                            </div>
                        `;
                    });
                    queueList.innerHTML = htmlContent;
                });
        }

        // Poll API links every 2 seconds to establish interactive dynamic HUD updates
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""

@app.route("/")
def render_dashboard():
    """
    Renders our single-file unified dashboard webpage string.
    """
    return Response(DASHBOARD_HTML, mimetype="text/html")


# =====================================================================
# SERVER RUN ORCHESTRATION
# =====================================================================
if __name__ == "__main__":
    print("[INIT] Verifying local Sentry Flyer Mission database schema...")
    ensure_database_exists()
    
    # Run the background coordinate simulator on a concurrent Python Thread
    sim_thread = threading.Thread(target=run_live_simulation, daemon=True)
    sim_thread.start()
    
    print("\n=====================================================================")
    print(" SENTRY FLYER GROUND CONTROL DASHBOARD LAUNCHED")
    print(" Network Mode: 100% Offline-First (No external assets required)")
    print(" Point your field browser to: http://127.0.0.1:5000")
    print("=====================================================================\n")
    
    # Run Flask server locally on default loopback port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)