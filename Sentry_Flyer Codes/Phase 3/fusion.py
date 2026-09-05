"""
Sentry Flyer — Unified Sensor Alignment & Cross-Modal Fusion Engine
Location: Phase 3/fusion.py
Purpose: Maps RGB bounding boxes into the thermal camera's pixel space via a
         calibrated homography and cross-checks radiometric temperature so
         a person or fire is only confirmed when both sensors agree.

Thermal input is expected as raw FLIR Lepton radiometric data: uint16
centi-Kelvin (value / 100 = Kelvin).

Why percentiles and contrast, not the mean
------------------------------------------
A bounding box around a person is mostly background (ground, rubble). The
mean of the box is therefore dominated by ambient temperature and would
reject almost every real person. We use a high percentile so the warmest
part of the box (skin, face, hands) drives the decision, and additionally
require that hot spot to stand out from the median temperature of the
*surroundings* (the box padded by CONTEXT_PAD_RATIO). A uniformly warm scene
(sun-heated rubble on a hot afternoon) therefore cannot be "verified" by
thermal alone; it stays "person_unverified" for a human to judge.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("sentry.fusion")

CENTI_KELVIN_OFFSET = 273.15


def centi_kelvin_to_celsius(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / 100.0 - CENTI_KELVIN_OFFSET


class SensorAligner:
    """Homography from thermal (160x120) pixel space into RGB (1920x1080) pixel space."""

    def __init__(self, thermal_src_pts=None, rgb_dst_pts=None):
        if thermal_src_pts is None or rgb_dst_pts is None:
            # Default: the two sensors are boresighted and share a field of view,
            # so the thermal corners map to the RGB corners.
            thermal_src_pts = [[0, 0], [160, 0], [160, 120], [0, 120]]
            rgb_dst_pts = [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]

        self.src_pts = np.asarray(thermal_src_pts, dtype=np.float32)
        self.dst_pts = np.asarray(rgb_dst_pts, dtype=np.float32)
        if len(self.src_pts) < 4 or len(self.src_pts) != len(self.dst_pts):
            raise ValueError("Need at least 4 matched calibration points")

        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)
        if self.H is None:
            raise ValueError("Calibration points are degenerate; homography failed")
        self.H_inv = np.linalg.inv(self.H)

    def warp_thermal_to_rgb(self, thermal_frame: np.ndarray, rgb_target_shape=(1080, 1920)) -> np.ndarray:
        """Resample the thermal frame onto the RGB pixel grid."""
        height, width = rgb_target_shape[:2]
        return cv2.warpPerspective(
            thermal_frame, self.H, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def rgb_points_to_thermal(self, points_xy: np.ndarray) -> np.ndarray:
        """Map an (N, 2) array of RGB pixel coordinates into thermal pixel coordinates."""
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)


class CrossModalFusion:
    # Skin-surface temperature as seen by a LWIR camera from altitude is well
    # below core body temperature. Bands are (verified, plausible) in Celsius.
    PERSON_VERIFIED_C = (28.0, 40.0)
    PERSON_PLAUSIBLE_C = (22.0, 45.0)
    PERSON_MIN_CONTRAST_C = 2.0  # hot spot must exceed the surrounding median by this much
    CONTEXT_PAD_RATIO = 0.5  # padding around the box (fraction of box size) used as "surroundings"
    FIRE_VERIFIED_C = 150.0
    FIRE_PLAUSIBLE_C = 80.0
    HOT_PERCENTILE = 95  # a person can be a handful of pixels in a 160x120 frame

    def __init__(
        self,
        thermal_w: int = 160,
        thermal_h: int = 120,
        aligner: Optional[SensorAligner] = None,
    ):
        self.thermal_w = thermal_w
        self.thermal_h = thermal_h
        self.aligner = aligner or SensorAligner()

    def project_box_to_thermal(self, rgb_box) -> list[int]:
        """RGB [x_min, y_min, x_max, y_max] -> clipped thermal-space box (inclusive)."""
        x_min, y_min, x_max, y_max = rgb_box
        corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        projected = self.aligner.rgb_points_to_thermal(corners)
        tx_min, ty_min = projected.min(axis=0)
        tx_max, ty_max = projected.max(axis=0)
        return [
            int(np.clip(np.floor(tx_min), 0, self.thermal_w - 1)),
            int(np.clip(np.floor(ty_min), 0, self.thermal_h - 1)),
            int(np.clip(np.ceil(tx_max), 0, self.thermal_w - 1)),
            int(np.clip(np.ceil(ty_max), 0, self.thermal_h - 1)),
        ]

    def _surrounding_median_c(self, frame_kelvin: np.ndarray, t_box: tuple[int, int, int, int]) -> float:
        """Median temperature of the box padded by CONTEXT_PAD_RATIO (clipped to the frame)."""
        x1, y1, x2, y2 = t_box
        pad_x = max(3, int((x2 - x1 + 1) * self.CONTEXT_PAD_RATIO))
        pad_y = max(3, int((y2 - y1 + 1) * self.CONTEXT_PAD_RATIO))
        h, w = frame_kelvin.shape[:2]
        context = frame_kelvin[max(0, y1 - pad_y) : min(h, y2 + pad_y + 1), max(0, x1 - pad_x) : min(w, x2 + pad_x + 1)]
        return float(np.median(centi_kelvin_to_celsius(context)))

    def verify_detection(self, rgb_class: str, rgb_box, thermal_frame_kelvin: np.ndarray, base_confidence: float):
        """
        Cross-check a visual detection against radiometric temperature.
        Returns (verified_class, confidence).
        """
        tx_min, ty_min, tx_max, ty_max = self.project_box_to_thermal(rgb_box)
        roi = thermal_frame_kelvin[ty_min : ty_max + 1, tx_min : tx_max + 1]
        if roi.size == 0:
            return rgb_class, round(base_confidence * 0.7, 2)

        temps_c = centi_kelvin_to_celsius(roi)

        if rgb_class == "person":
            hot = float(np.percentile(temps_c, self.HOT_PERCENTILE))
            contrast = hot - self._surrounding_median_c(thermal_frame_kelvin, (tx_min, ty_min, tx_max, ty_max))
            in_verified_band = self.PERSON_VERIFIED_C[0] <= hot <= self.PERSON_VERIFIED_C[1]
            in_plausible_band = self.PERSON_PLAUSIBLE_C[0] <= hot <= self.PERSON_PLAUSIBLE_C[1]
            if in_verified_band and contrast >= self.PERSON_MIN_CONTRAST_C:
                return "person", round(min(base_confidence * 1.15, 0.98), 2)
            if in_plausible_band:
                return "person_unverified", round(base_confidence * 0.85, 2)
            return "environmental_anomaly", round(base_confidence * 0.30, 2)

        if rgb_class == "fire":
            peak = float(temps_c.max())
            if peak > self.FIRE_VERIFIED_C:
                return "fire", round(max(base_confidence, 0.95), 2)
            if peak > self.FIRE_PLAUSIBLE_C:
                return "thermal_anomaly", round(base_confidence * 0.75, 2)
            return "reflector_false_alarm", round(base_confidence * 0.20, 2)

        # Smoke, floodwater, damage: thermal adds no discriminating signal.
        return rgb_class, round(base_confidence, 2)


# =====================================================================
# SYSTEM INTEGRATION TEST
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("[TEST] Initializing Sentry Flyer Unified Fusion Engine...")
    fusion_engine = CrossModalFusion()

    # 1. Alignment layer
    mock_thermal_raw = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(mock_thermal_raw, (80, 60), 10, 255, -1)
    warped = fusion_engine.aligner.warp_thermal_to_rgb(mock_thermal_raw)
    print(f"[TEST] Alignment successful. Warped array shape: {warped.shape}")

    # 2. Verification layer. Ambient 25 C everywhere, a 34 C body where the RGB box lands.
    ambient_ck = int((25.0 + CENTI_KELVIN_OFFSET) * 100)
    body_ck = int((34.0 + CENTI_KELVIN_OFFSET) * 100)
    mock_thermal_kelvin = np.full((120, 160), ambient_ck, dtype=np.uint16)

    pixel_box = [540, 405, 660, 495]  # RGB box -> thermal ~[45, 45, 55, 55]
    tb = fusion_engine.project_box_to_thermal(pixel_box)
    mock_thermal_kelvin[tb[1] : tb[3] + 1, tb[0] : tb[2] + 1] = body_ck

    v_class, v_conf = fusion_engine.verify_detection("person", pixel_box, mock_thermal_kelvin, 0.85)

    print("\n=== SENSOR AGREEMENT TEST REPORT ===")
    print(f"Thermal ROI:   {tb}")
    print("Input Target:  Class='person', Conf=0.85")
    print(f"Fused Target:  Class='{v_class}', Conf={v_conf}")
    print(f"Status:        {'TARGET VERIFIED' if v_class == 'person' else 'REJECTED FALSE ALARM'}")
    print("====================================\n")
