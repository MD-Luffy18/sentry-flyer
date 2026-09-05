"""
Unit tests for the Sentry Flyer pipeline modules.

The "Phase N" folders are not importable packages, so each module is loaded
by file path. Run with:  pytest tests/
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent / "Sentry_Flyer Codes"


def load(phase: str, name: str):
    path = ROOT / phase / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


database = load("Phase 1", "database")
planner = load("Phase 1", "planner")
odometry = load("Phase 2", "odometry")
projection = load("Phase 2", "projection")
detect = load("Phase 3", "detect")
fusion = load("Phase 3", "fusion")
report_generator = load("Phase 4", "report_generator")


# ---------------------------------------------------------------------------
# Phase 1: database + clustering
# ---------------------------------------------------------------------------
def test_haversine_known_distance():
    # 0.001 deg of latitude is ~111 m.
    assert database.haversine_m(25.31, 78.488, 25.311, 78.488) == pytest.approx(111.2, abs=0.5)
    assert database.haversine_m(0, 0, 0, 0) == 0.0


def test_clustering_merges_nearby_and_splits_far(tmp_path):
    db = database.SentryDatabase(tmp_path / "m.db")
    c1 = db.insert_and_cluster_detection("person", 25.31000, 78.48800, 0.85, 95, "Fused")
    c2 = db.insert_and_cluster_detection("person", 25.31004, 78.48804, 0.95, 90, "Fused")  # ~6 m
    c3 = db.insert_and_cluster_detection("person", 25.31040, 78.48840, 0.80, 95, "Fused")  # ~60 m
    c4 = db.insert_and_cluster_detection("fire", 25.31000, 78.48800, 0.80, 80, "RGB")  # same spot, other class

    assert c1 == c2
    assert c3 != c1 and c4 != c1

    rows = {r["id"]: r for r in db.fetch_prioritized_map()}
    merged = rows[c1]
    assert merged["detection_count"] == 2
    assert merged["mean_confidence"] == pytest.approx(0.90)
    assert merged["max_severity"] == 95
    assert merged["center_latitude"] == pytest.approx(25.31002)
    # Priority ordering: severity * confidence, descending.
    scores = [r["priority_score"] for r in db.fetch_prioritized_map()]
    assert scores == sorted(scores, reverse=True)


def test_raw_detection_links_to_cluster(tmp_path):
    db = database.SentryDatabase(tmp_path / "m.db")
    cid = db.insert_and_cluster_detection("smoke", 25.31, 78.488, 0.6, 40, "RGB", "f.png")
    with db._connection() as conn:
        row = conn.execute("SELECT cluster_id, frame_filename FROM raw_detections").fetchone()
    assert row["cluster_id"] == cid and row["frame_filename"] == "f.png"


def test_telemetry_roundtrip(tmp_path):
    db = database.SentryDatabase(tmp_path / "m.db")
    assert db.latest_telemetry() is None
    db.log_telemetry(1, 2, 3, 4, 5, 6, 7, timestamp=100.0)
    db.log_telemetry(1.5, 2, 3, 4, 5, 6, 7, timestamp=101.0)
    assert db.latest_telemetry()["latitude"] == 1.5


# ---------------------------------------------------------------------------
# Phase 1: planner (pure functions only; MAVLink link needs hardware)
# ---------------------------------------------------------------------------
def test_lawnmower_alternates_direction():
    wps = planner.generate_lawnmower_waypoints(0.0, 0.0, 0.004, 0.01, rows=3)
    assert wps == [(0.0, 0.0), (0.0, 0.01), (0.002, 0.01), (0.002, 0.0), (0.004, 0.0), (0.004, 0.01)]


def test_lawnmower_single_row_and_swath():
    assert planner.generate_lawnmower_waypoints(1.0, 2.0, 1.0, 3.0, rows=1) == [(1.0, 2.0), (1.0, 3.0)]
    # 111 m tall box with 30 m swath -> 5 rows -> 10 waypoints.
    assert len(planner.generate_lawnmower_waypoints(0.0, 0.0, 0.001, 0.001, swath_m=30.0)) == 10
    with pytest.raises(ValueError):
        planner.generate_lawnmower_waypoints(0, 0, 1, 1, rows=0)


# ---------------------------------------------------------------------------
# Phase 2: odometry + projection
# ---------------------------------------------------------------------------
def _frame(shift_px: int) -> np.ndarray:
    import cv2

    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    for cx, cy in ((110, 110), (510, 110), (110, 360), (510, 360)):
        x0 = cx - 10 + shift_px
        cv2.rectangle(frame, (x0, cy - 10), (x0 + 20, cy + 10), (0, 0, 0), -1)
    return frame


def test_odometry_sign_and_scale():
    vo = odometry.VisualOdometry(h_fov_deg=100.0)
    vo.initialize_tracking(_frame(0))
    right, forward = vo.update_odometry(_frame(-8), current_altitude=20.0)
    mpp = 2 * 20.0 * math.tan(math.radians(50)) / 640
    assert right == pytest.approx(8 * mpp, rel=0.05)  # ground moved left => aircraft moved right
    assert abs(forward) < 0.05
    assert vo.accumulated_x == pytest.approx(right)


def test_odometry_reseeds_when_too_few_features():
    import cv2

    # Two blocks give at most 8 corners, below MIN_TRACKED_POINTS, so the
    # update must re-seed and report no motion rather than guess.
    sparse = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(sparse, (100, 100), (120, 120), (0, 0, 0), -1)
    cv2.rectangle(sparse, (500, 350), (520, 370), (0, 0, 0), -1)
    vo = odometry.VisualOdometry()
    vo.initialize_tracking(sparse)
    assert vo.update_odometry(sparse, 20.0) == (0.0, 0.0)
    assert vo.accumulated_x == 0.0


def test_projection_center_pixel_geometry():
    p = projection.CoordinateProjector(1920, 1080, h_fov=100.0)
    # 50 m up, 45 deg down, looking north: ground point 50 m north.
    lat, lon = p.project_pixel_to_gps(25.31, 78.488, 50.0, 0.0, -45.0, 960, 540)
    assert lat == pytest.approx(25.31 + 50 / 111111, abs=1e-6)
    assert lon == pytest.approx(78.488, abs=1e-6)
    # Facing east, same geometry: shift in longitude only.
    lat_e, lon_e = p.project_pixel_to_gps(25.31, 78.488, 50.0, 90.0, -45.0, 960, 540)
    assert lat_e == pytest.approx(25.31, abs=1e-6) and lon_e > 78.488


def test_projection_vertical_fov_uses_tangent_ratio():
    p = projection.CoordinateProjector(1920, 1080, h_fov=100.0)
    expected = math.degrees(2 * math.atan(math.tan(math.radians(50)) * 1080 / 1920))
    assert p.v_fov == pytest.approx(expected)
    assert p.v_fov != pytest.approx(100.0 * 1080 / 1920)  # the old linear approximation


def test_projection_clamps_near_horizon():
    p = projection.CoordinateProjector(min_depression_deg=5.0)
    lat, _ = p.project_pixel_to_gps(0.0, 0.0, 10.0, 0.0, 0.0, 960, 540)
    assert lat == pytest.approx((10 / math.tan(math.radians(5))) / 111111, abs=1e-6)
    with pytest.raises(ValueError):
        p.project_pixel_to_gps(0, 0, 0.0, 0, -45, 0, 0)


# ---------------------------------------------------------------------------
# Phase 3: detector output decoding + fusion
# ---------------------------------------------------------------------------
def test_decode_yolov5_layout():
    d = detect.EdgeDetector(conf_threshold=0.5)
    nc = len(d.classes)
    rows = np.zeros((3, 5 + nc), dtype=np.float32)
    rows[0, :5] = [100, 100, 40, 80, 0.9]; rows[0, 5] = 0.95  # person, 0.855
    rows[1, :5] = [200, 200, 40, 80, 0.9]; rows[1, 6] = 0.4   # fire, 0.36 -> dropped
    rows[2, :5] = [300, 300, 40, 80, 0.8]; rows[2, 9] = 0.9   # damage, 0.72
    boxes, scores, ids = d._decode(rows[None, ...])
    assert ids.tolist() == [0, 4]
    assert scores == pytest.approx([0.855, 0.72])
    assert boxes[0].tolist() == [100, 100, 40, 80]


def test_decode_yolov8_transposed_layout():
    d = detect.EdgeDetector(conf_threshold=0.5)
    nc = len(d.classes)
    cols = np.zeros((4 + nc, 2), dtype=np.float32)
    cols[:4, 0] = [50, 60, 10, 20]; cols[4 + 1, 0] = 0.8  # fire
    cols[:4, 1] = [70, 80, 10, 20]; cols[4 + 2, 1] = 0.2  # smoke, dropped
    boxes, scores, ids = d._decode(cols[None, ...])
    assert ids.tolist() == [1] and scores == pytest.approx([0.8])
    assert boxes[0].tolist() == [50, 60, 10, 20]


def test_simulation_mode_when_no_model():
    d = detect.EdgeDetector()
    dets, latency = d.run_inference(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert d.simulation_mode and dets[0]["class"] == "person" and latency > 0


def _thermal(ambient_c: float, patch_c: float | None, box: list[int]) -> np.ndarray:
    frame = np.full((120, 160), int((ambient_c + 273.15) * 100), dtype=np.uint16)
    if patch_c is not None:
        x1, y1, x2, y2 = box
        frame[y1 : y2 + 1, x1 : x2 + 1] = int((patch_c + 273.15) * 100)
    return frame


def test_fusion_box_projection_default_scale():
    f = fusion.CrossModalFusion()
    assert f.project_box_to_thermal([540, 405, 660, 495]) == [45, 45, 55, 55]
    assert f.project_box_to_thermal([-100, -100, 5000, 5000]) == [0, 0, 159, 119]


def test_fusion_person_verified_rejected_and_unverified():
    f = fusion.CrossModalFusion()
    box = [540, 405, 660, 495]
    tb = f.project_box_to_thermal(box)
    assert f.verify_detection("person", box, _thermal(25, 34, tb), 0.85) == ("person", 0.98)
    # Cold box: nothing warm at all -> anomaly.
    assert f.verify_detection("person", box, _thermal(10, None, tb), 0.85) == ("environmental_anomaly", 0.26)
    # Plausible but too hot for skin -> unverified.
    assert f.verify_detection("person", box, _thermal(25, 43, tb), 0.85)[0] == "person_unverified"
    # Uniformly body-warm box (sun-heated rubble): no contrast, so thermal cannot confirm.
    assert f.verify_detection("person", box, _thermal(34, None, tb), 0.85)[0] == "person_unverified"


def test_fusion_person_ignores_cold_background():
    # Only a small part of the box is warm; the mean would fail, the percentile passes.
    f = fusion.CrossModalFusion()
    box = [540, 405, 660, 495]
    frame = _thermal(20, None, [0, 0, 0, 0])
    frame[50:54, 50:54] = int((33.0 + 273.15) * 100)  # 16 of 121 ROI pixels
    assert f.verify_detection("person", box, frame, 0.8)[0] == "person"


def test_fusion_fire_thresholds():
    f = fusion.CrossModalFusion()
    box = [0, 0, 100, 100]
    tb = f.project_box_to_thermal(box)
    assert f.verify_detection("fire", box, _thermal(25, 300, tb), 0.6) == ("fire", 0.95)
    assert f.verify_detection("fire", box, _thermal(25, 100, tb), 0.6) == ("thermal_anomaly", 0.45)
    assert f.verify_detection("fire", box, _thermal(25, 30, tb), 0.6) == ("reflector_false_alarm", 0.12)


def test_fusion_custom_calibration_uses_homography():
    # Thermal covers only the centre half of the RGB frame.
    aligner = fusion.SensorAligner(
        thermal_src_pts=[[0, 0], [160, 0], [160, 120], [0, 120]],
        rgb_dst_pts=[[480, 270], [1440, 270], [1440, 810], [480, 810]],
    )
    f = fusion.CrossModalFusion(aligner=aligner)
    assert f.project_box_to_thermal([480, 270, 1440, 810]) == [0, 0, 159, 119]


# ---------------------------------------------------------------------------
# Phase 4: reporting
# ---------------------------------------------------------------------------
def _seeded_db(tmp_path, n: int):
    db = database.SentryDatabase(tmp_path / "m.db")
    for i in range(n):
        db.insert_and_cluster_detection("person", 25.31 + i * 0.001, 78.488, 0.9, 50 + i, "Fused")
    return db


def test_geojson_report(tmp_path):
    _seeded_db(tmp_path, 2)
    rep = report_generator.MissionReporter(tmp_path / "m.db")
    data = rep.build_geojson()
    assert data["type"] == "FeatureCollection" and len(data["features"]) == 2
    feat = data["features"][0]
    assert feat["geometry"]["coordinates"] == [78.488, pytest.approx(25.311)]  # highest severity first
    assert feat["properties"]["priority_score"] >= data["features"][1]["properties"]["priority_score"]
    out = tmp_path / "r.geojson"
    assert rep.export_geojson_report(out) and out.exists()


def test_lora_packet_checksum_and_truncation(tmp_path):
    _seeded_db(tmp_path, 12)
    rep = report_generator.MissionReporter(tmp_path / "m.db")
    full = rep.build_lora_packet(max_bytes=10_000)
    assert full.startswith("$SF,12,") and report_generator.verify_lora_packet(full)

    small = rep.build_lora_packet(max_bytes=120)
    assert len(small.encode("ascii")) <= 120
    assert report_generator.verify_lora_packet(small)
    count = int(small.split(",")[1])
    assert 0 < count < 12
    # Highest-severity target (last inserted) is kept.
    assert ":61" in small

    assert not report_generator.verify_lora_packet(full[:-1] + "0")
    assert report_generator.verify_lora_packet("$SF,0,EMPTY*" + report_generator.nmea_checksum("SF,0,EMPTY"))


def test_reporter_missing_db_is_graceful(tmp_path):
    rep = report_generator.MissionReporter(tmp_path / "nope.db")
    assert rep.export_geojson_report(tmp_path / "x.geojson") is False
    assert rep.generate_micro_lora_packet(tmp_path / "x.txt") is None
