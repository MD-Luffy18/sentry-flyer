"""
Sentry Flyer — Ground Station Dashboard & Telemetry Link
Location: Phase 5/app.py
Purpose: Offline-first Flask server presenting a live incident map, telemetry
         readouts, and a prioritised triage queue for responders. No external
         assets: the map is drawn on a <canvas> from the mission database.

Configuration (environment variables)
-------------------------------------
SENTRY_DB        Path to the mission SQLite database (default: Phase 1/sentry_mission.db)
SENTRY_HOST      Bind address (default 127.0.0.1; use 0.0.0.0 to serve a field LAN)
SENTRY_PORT      Port (default 5000)
SENTRY_SIMULATE  "1" to run the built-in flight simulator (default 1). Set to 0
                 when the real drone is writing to the database.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from flask import Flask, Response, jsonify, request

log = logging.getLogger("sentry.dashboard")

HERE = Path(__file__).resolve().parent
PHASE1_DIR = HERE.parent / "Phase 1"
SCHEMA_PATH = PHASE1_DIR / "schema.sql"
DATABASE_PATH = Path(os.environ.get("SENTRY_DB", PHASE1_DIR / "sentry_mission.db"))

MISSION_ORIGIN = (25.31000, 78.48800)  # Default map centre until telemetry arrives

app = Flask(__name__)


# =====================================================================
# DATABASE
# =====================================================================
@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DATABASE_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_database_exists() -> None:
    """Create the database from the shared Phase 1 schema if needed."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with db_connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(schema)


def _load_reporter():
    """Import Phase 4's MissionReporter by path (the folder names are not importable packages)."""
    spec = importlib.util.spec_from_file_location("report_generator", HERE.parent / "Phase 4" / "report_generator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MissionReporter(DATABASE_PATH)


# =====================================================================
# SIMULATOR (demo only)
# =====================================================================
def run_live_simulation(period_s: float = 4.0) -> None:
    """Insert a slow diagonal flight path and two detections so the UI has something to show."""
    log.info("Starting live mission simulation thread")
    ensure_database_exists()

    step = 0
    while True:
        try:
            now = time.time()
            sim_lat = MISSION_ORIGIN[0] + step * 0.00015
            sim_lon = MISSION_ORIGIN[1] + step * 0.00010
            sim_alt = 45.0 + (step % 3)
            sim_volt = max(12.6 - step * 0.05, 10.8)

            with db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO telemetry_logs
                        (latitude, longitude, altitude, heading, pitch, roll, battery_voltage, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sim_lat, sim_lon, sim_alt, 33.7, 1.2, -0.4, sim_volt, now),
                )
                detection = {
                    2: ("person", 25.31050, 78.48850, 0.94, 95, 3),
                    5: ("fire", 25.31120, 78.48910, 0.88, 90, 1),
                }.get(step)
                if detection:
                    log.info("Simulated detection: %s", detection[0])
                    conn.execute(
                        """
                        INSERT INTO fused_clusters
                            (class_name, center_latitude, center_longitude, mean_confidence,
                             max_severity, detection_count, first_seen, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (*detection, now, now),
                    )
        except Exception as e:  # noqa: BLE001 - keep the demo thread alive
            log.warning("Simulator loop skipped: %s", e)

        step += 1
        time.sleep(period_s)


# =====================================================================
# API
# =====================================================================
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "database": str(DATABASE_PATH), "time": time.time()})


@app.route("/api/telemetry")
def get_telemetry():
    """Latest telemetry row, or a placeholder at the mission origin."""
    try:
        with db_connection() as conn:
            row = conn.execute("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT 1").fetchone()
    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500

    if row is None:
        return jsonify(
            {
                "latitude": MISSION_ORIGIN[0],
                "longitude": MISSION_ORIGIN[1],
                "altitude": 0.0,
                "heading": 0.0,
                "battery_voltage": 0.0,
                "timestamp": None,
                "live": False,
            }
        )
    return jsonify(
        {
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "altitude": round(row["altitude"], 1),
            "heading": round(row["heading"], 1),
            "battery_voltage": round(row["battery_voltage"], 2),
            "timestamp": row["timestamp"],
            "live": True,
        }
    )


@app.route("/api/track")
def get_track():
    """Recent flight path as [[lat, lon], ...], oldest first."""
    limit = min(int(request.args.get("limit", 300)), 5000)
    try:
        with db_connection() as conn:
            rows = conn.execute(
                "SELECT latitude, longitude FROM telemetry_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500
    return jsonify([[r["latitude"], r["longitude"]] for r in reversed(rows)])


@app.route("/api/targets")
def get_targets():
    """All clusters sorted by priority score (severity x confidence)."""
    try:
        with db_connection() as conn:
            rows = conn.execute(
                """
                SELECT *, (max_severity * mean_confidence) AS priority_score
                FROM fused_clusters
                ORDER BY priority_score DESC, last_updated DESC
                """
            ).fetchall()
    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(
        [
            {
                "id": r["id"],
                "class_name": r["class_name"],
                "latitude": r["center_latitude"],
                "longitude": r["center_longitude"],
                "confidence": round(r["mean_confidence"], 2),
                "severity": r["max_severity"],
                "count": r["detection_count"],
                "priority_score": round(r["priority_score"], 1),
                "last_updated": r["last_updated"],
            }
            for r in rows
        ]
    )


@app.route("/api/report.geojson")
def get_geojson_report():
    """Download the current situation report as GeoJSON."""
    try:
        data = _load_reporter().build_geojson()
    except (FileNotFoundError, sqlite3.Error) as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(data), 200, {"Content-Disposition": "attachment; filename=mission_report.geojson"}


# =====================================================================
# DASHBOARD FRONTEND (single embedded page, zero external assets)
# =====================================================================
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentry Flyer — Responder Command Center</title>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --border: #334155;
    --text: #f8fafc; --muted: #94a3b8;
    --primary: #3b82f6; --critical: #ef4444; --warning: #f59e0b; --ok: #10b981;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  }
  header {
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 12px 24px; display: flex; justify-content: space-between; align-items: center;
  }
  header h1 { font-size: 18px; font-weight: 700; letter-spacing: 0.5px; }
  header .status { display: flex; align-items: center; gap: 12px; font-size: 13px; color: var(--muted); }
  header a { color: var(--primary); text-decoration: none; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 8px var(--ok); }
  .dot.stale { background: var(--warning); box-shadow: 0 0 8px var(--warning); }
  .main { display: flex; flex: 1; min-height: 0; }
  .sidebar { width: 380px; border-right: 1px solid var(--border); background: var(--panel); display: flex; flex-direction: column; }
  .telemetry { padding: 16px; border-bottom: 1px solid var(--border); display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .card { background: rgba(15,23,42,.4); border: 1px solid var(--border); padding: 10px 12px; border-radius: 6px; }
  .card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }
  .card .value { font-size: 16px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .triage { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .triage-header { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; }
  .triage-list { flex: 1; overflow-y: auto; padding: 12px; }
  .item { background: rgba(15,23,42,.3); border: 1px solid var(--border); border-left: 4px solid var(--primary); border-radius: 4px; padding: 10px 12px; margin-bottom: 8px; }
  .item.critical { border-left-color: var(--critical); }
  .item.warning { border-left-color: var(--warning); }
  .item-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
  .item-header .class { font-weight: 600; font-size: 14px; text-transform: capitalize; }
  .item-header .score { font-size: 11px; background: rgba(255,255,255,.1); padding: 2px 6px; border-radius: 3px; color: var(--muted); }
  .item-details { font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 2px; font-variant-numeric: tabular-nums; }
  .empty { text-align: center; padding: 20px; color: var(--muted); font-size: 13px; }
  .map { flex: 1; position: relative; background: #0c111d; }
  canvas { display: block; width: 100%; height: 100%; }
  .legend { position: absolute; left: 12px; bottom: 12px; font-size: 11px; color: var(--muted); background: rgba(15,23,42,.7); padding: 8px 10px; border-radius: 4px; line-height: 1.6; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
</style>
</head>
<body>
<header>
  <h1>SENTRY FLYER — GROUND CONTROL</h1>
  <div class="status">
    <a href="/api/report.geojson">Download GeoJSON</a>
    <div class="dot" id="link-dot"></div>
    <span id="link-text">WAITING FOR TELEMETRY</span>
  </div>
</header>

<div class="main">
  <div class="sidebar">
    <div class="telemetry">
      <div class="card"><div class="label">Drone Position</div><div class="value" id="t-coords">---, ---</div></div>
      <div class="card"><div class="label">Alt (Baro)</div><div class="value" id="t-alt">0.0 m</div></div>
      <div class="card"><div class="label">Compass</div><div class="value" id="t-heading">0.0°</div></div>
      <div class="card"><div class="label">Battery</div><div class="value" id="t-battery">0.0 V</div></div>
    </div>
    <div class="triage">
      <div class="triage-header">Prioritized Incident Queue</div>
      <div class="triage-list" id="queue"><div class="empty">Scanning search sector...</div></div>
    </div>
  </div>

  <div class="map">
    <canvas id="map"></canvas>
    <div class="legend">
      <div><i style="background:#ef4444"></i>Person</div>
      <div><i style="background:#f59e0b"></i>Fire / thermal</div>
      <div><i style="background:#3b82f6"></i>Flood / smoke / damage</div>
      <div><i style="background:#10b981"></i>Drone &amp; track</div>
    </div>
  </div>
</div>

<script>
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  const CLASS_COLOR = cls => {
    if (cls.startsWith('person')) return '#ef4444';
    if (cls === 'fire' || cls === 'thermal_anomaly') return '#f59e0b';
    return '#3b82f6';
  };

  const state = { telemetry: null, track: [], targets: [] };

  // ---- Local flat-earth projection around the drone ----------------------
  const M_PER_DEG_LAT = 111111;
  function toLocal(lat, lon, origin) {
    const mLon = M_PER_DEG_LAT * Math.cos(origin[0] * Math.PI / 180);
    return [(lon - origin[1]) * mLon, (lat - origin[0]) * M_PER_DEG_LAT]; // [east, north] metres
  }

  function drawMap() {
    const canvas = $('map');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const t = state.telemetry;
    const origin = t ? [t.latitude, t.longitude] : [25.31, 78.488];
    const pts = [
      ...state.track.map(p => toLocal(p[0], p[1], origin)),
      ...state.targets.map(p => toLocal(p.latitude, p.longitude, origin)),
      [0, 0]
    ];
    const extent = Math.max(120, ...pts.map(p => Math.hypot(p[0], p[1]))) * 1.15;
    const scale = Math.min(w, h) / 2 / extent;
    const cx = w / 2, cy = h / 2;
    const px = ([e, n]) => [cx + e * scale, cy - n * scale];

    // Range rings every 100 m
    ctx.strokeStyle = '#1f2a44'; ctx.fillStyle = '#475569'; ctx.font = '11px system-ui'; ctx.lineWidth = 1;
    for (let r = 100; r < extent; r += 100) {
      ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, Math.PI * 2); ctx.stroke();
      ctx.fillText(r + ' m', cx + r * scale + 4, cy - 4);
    }

    // Flight track
    if (state.track.length > 1) {
      ctx.strokeStyle = 'rgba(16,185,129,.6)'; ctx.lineWidth = 2; ctx.beginPath();
      state.track.forEach((p, i) => { const [x, y] = px(toLocal(p[0], p[1], origin)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }

    // Targets
    state.targets.forEach(tg => {
      const [x, y] = px(toLocal(tg.latitude, tg.longitude, origin));
      const r = 6 + Math.min(6, tg.count);
      ctx.fillStyle = CLASS_COLOR(tg.class_name);
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#0c111d'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#e2e8f0'; ctx.font = '12px system-ui';
      ctx.fillText(tg.class_name.replace(/_/g, ' ') + ' (' + tg.priority_score + ')', x + r + 4, y + 4);
    });

    // Drone: triangle pointing along heading
    const hdg = (t ? t.heading : 0) * Math.PI / 180;
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(hdg);
    ctx.fillStyle = '#10b981'; ctx.beginPath();
    ctx.moveTo(0, -12); ctx.lineTo(8, 10); ctx.lineTo(0, 5); ctx.lineTo(-8, 10); ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  // ---- Data polling ------------------------------------------------------
  const getJSON = url => fetch(url).then(r => r.json());

  function renderTelemetry(d) {
    if (d.error) return;
    state.telemetry = d;
    $('t-coords').textContent = d.latitude.toFixed(5) + ', ' + d.longitude.toFixed(5);
    $('t-alt').textContent = d.altitude.toFixed(1) + ' m';
    $('t-heading').textContent = d.heading.toFixed(1) + '°';
    $('t-battery').textContent = d.battery_voltage.toFixed(2) + ' V';
    const age = d.timestamp ? (Date.now() / 1000 - d.timestamp) : Infinity;
    const stale = !d.live || age > 15;
    $('link-dot').classList.toggle('stale', stale);
    $('link-text').textContent = stale ? 'TELEMETRY STALE' : 'OFFLINE TACTICAL NETWORK · LIVE';
  }

  function renderTargets(targets) {
    if (targets.error) return;
    state.targets = targets;
    const list = $('queue');
    if (!targets.length) { list.innerHTML = '<div class="empty">No targets detected in search pattern yet.</div>'; return; }
    list.innerHTML = targets.map(t => `
      <div class="item ${t.priority_score >= 80 ? 'critical' : 'warning'}">
        <div class="item-header">
          <span class="class">${esc(t.class_name.replace(/_/g, ' '))}</span>
          <span class="score">Score: ${t.priority_score}</span>
        </div>
        <div class="item-details">
          <span><strong>Coordinates:</strong> ${t.latitude.toFixed(5)}, ${t.longitude.toFixed(5)}</span>
          <span><strong>Confidence:</strong> ${(t.confidence * 100).toFixed(0)}% &nbsp;|&nbsp; <strong>Severity:</strong> ${t.severity} &nbsp;|&nbsp; <strong>Sightings:</strong> ${t.count}</span>
        </div>
      </div>`).join('');
  }

  async function update() {
    try {
      const [tel, targets, track] = await Promise.all([getJSON('/api/telemetry'), getJSON('/api/targets'), getJSON('/api/track')]);
      renderTelemetry(tel);
      renderTargets(targets);
      if (!track.error) state.track = track;
      drawMap();
    } catch (e) {
      $('link-dot').classList.add('stale');
      $('link-text').textContent = 'GROUND STATION UNREACHABLE';
    }
  }

  window.addEventListener('resize', drawMap);
  setInterval(update, 2000);
  update();
</script>
</body>
</html>
"""


@app.route("/")
def render_dashboard():
    return Response(DASHBOARD_HTML, mimetype="text/html")


# =====================================================================
# SERVER RUN ORCHESTRATION
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    host = os.environ.get("SENTRY_HOST", "127.0.0.1")
    port = int(os.environ.get("SENTRY_PORT", "5000"))
    simulate = os.environ.get("SENTRY_SIMULATE", "1") == "1"

    log.info("Verifying mission database schema at %s", DATABASE_PATH)
    ensure_database_exists()

    if simulate:
        threading.Thread(target=run_live_simulation, daemon=True, name="simulator").start()
    else:
        log.info("Simulator disabled; expecting the drone to write telemetry")

    print("\n=====================================================================")
    print(" SENTRY FLYER GROUND CONTROL DASHBOARD LAUNCHED")
    print(" Network Mode: 100% Offline-First (No external assets required)")
    print(f" Point your field browser to: http://{host}:{port}")
    print("=====================================================================\n")
    app.run(host=host, port=port, debug=False, threaded=True)
