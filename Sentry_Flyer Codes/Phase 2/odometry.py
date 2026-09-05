"""
Sentry Flyer — Visual Odometry GPS-Denied Fallback Engine
Location: Phase 2/odometry.py
Purpose: Tracks high-contrast ground features across video frames using
         Lucas-Kanade optical flow and converts the observed pixel motion
         into a body-frame displacement estimate when GPS is unavailable.

Conventions
-----------
* Camera is nadir (pointing straight down) with image "up" aligned to the
  aircraft's forward axis.
* update_odometry() returns (right_m, forward_m): metres the *aircraft*
  moved, which is the negative of the direction the ground appears to move.
* Scale is derived from altitude and horizontal field of view:
  metres_per_pixel = 2 * alt * tan(h_fov / 2) / frame_width.
  Pass `meters_per_pixel_at_1m` to override with a calibrated value.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("sentry.odometry")


class VisualOdometry:
    MIN_TRACKED_POINTS = 10  # below this we re-seed the feature grid

    def __init__(
        self,
        max_features: int = 100,
        h_fov_deg: float = 100.0,
        meters_per_pixel_at_1m: Optional[float] = None,
        replenish_ratio: float = 0.6,
    ):
        self.max_features = max_features
        self.h_fov_rad = math.radians(h_fov_deg)
        self._mpp_override = meters_per_pixel_at_1m
        self.replenish_ratio = replenish_ratio

        # Shi-Tomasi corner detection: reliable, repeatable tracking points.
        self.feature_params = dict(maxCorners=max_features, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Lucas-Kanade pyramidal optical flow.
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.accumulated_x = 0.0  # right (+) / left (-), metres
        self.accumulated_y = 0.0  # forward (+) / back (-), metres
        self.last_frame_time: Optional[float] = None

    # ------------------------------------------------------------------ #
    def meters_per_pixel(self, altitude_m: float, frame_width_px: int) -> float:
        """Ground distance covered by one pixel at the given altitude."""
        if self._mpp_override is not None:
            return altitude_m * self._mpp_override
        ground_width_m = 2.0 * altitude_m * math.tan(self.h_fov_rad / 2.0)
        return ground_width_m / frame_width_px

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)

    def initialize_tracking(self, frame: np.ndarray) -> None:
        """Seed the tracker from the first frame."""
        self.prev_gray = self._to_gray(frame)
        self.prev_points = self._detect(self.prev_gray)
        self.last_frame_time = time.time()
        if self.prev_points is not None:
            log.info("Locked onto %d terrain features", len(self.prev_points))
        else:
            log.warning("Low-texture terrain: no trackable features found")

    def reset(self) -> None:
        """Forget tracking state and accumulated drift (e.g. when GPS returns)."""
        self.prev_gray = None
        self.prev_points = None
        self.accumulated_x = 0.0
        self.accumulated_y = 0.0
        self.last_frame_time = None

    # ------------------------------------------------------------------ #
    def update_odometry(self, current_frame: np.ndarray, current_altitude: float) -> tuple[float, float]:
        """
        Estimate aircraft displacement since the previous frame.

        Returns (right_m, forward_m). Returns (0, 0) when tracking must be
        (re)seeded, so callers can always add the result to a running estimate.
        """
        gray = self._to_gray(current_frame)
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
            self.initialize_tracking(gray)
            return 0.0, 0.0

        now = time.time()
        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if next_points is None or status is None:
            tracked = np.zeros(0, dtype=bool)
        else:
            tracked = status.ravel() == 1

        if tracked.sum() < self.MIN_TRACKED_POINTS:
            log.warning("Tracking points depleted (%d); re-seeding feature grid", int(tracked.sum()))
            self.prev_gray = gray
            self.prev_points = self._detect(gray)
            self.last_frame_time = now
            return 0.0, 0.0

        good_new = next_points[tracked].reshape(-1, 2)
        good_old = self.prev_points[tracked].reshape(-1, 2)

        # Median is robust to the odd feature that latched onto moving debris.
        flow = np.median(good_new - good_old, axis=0)
        mpp = self.meters_per_pixel(current_altitude, gray.shape[1])

        # Ground moving left in the image means the aircraft moved right;
        # ground moving down (+y in image coords) means the aircraft moved forward.
        right_m = float(-flow[0] * mpp)
        forward_m = float(flow[1] * mpp)

        self.accumulated_x += right_m
        self.accumulated_y += forward_m

        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2).astype(np.float32)
        self.last_frame_time = now

        # Top up the feature set when it has thinned, masking out existing points
        # so we do not track the same corner twice.
        if len(self.prev_points) < self.max_features * self.replenish_ratio:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            for x, y in self.prev_points.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), self.feature_params["minDistance"], 0, -1)
            fresh = self._detect(gray, mask)
            if fresh is not None:
                self.prev_points = np.vstack((self.prev_points, fresh))[: self.max_features]

        return right_m, forward_m


# =====================================================================
# INTEGRATION TESTING RUN (Executable validation)
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("[TEST] Initializing Sentry Flyer Visual Odometry Tracker...")
    vo = VisualOdometry()

    def make_frame(shift_px: int) -> np.ndarray:
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        for cx, cy in ((110, 110), (510, 110), (110, 360), (510, 360)):
            x0 = cx - 10 + shift_px
            cv2.rectangle(frame, (x0, cy - 10), (x0 + 20, cy + 10), (0, 0, 0), -1)
        return frame

    vo.initialize_tracking(make_frame(0))
    # Ground features move left 8 px, so the aircraft drifted right.
    dx, dy = vo.update_odometry(make_frame(-8), current_altitude=20.0)

    print("\n=== VISUAL ODOMETRY CALCULATOR ===")
    print(f"Metres per pixel @20 m: {vo.meters_per_pixel(20.0, 640):.4f}")
    print(f"Calculated Shift Right:   {dx:+.4f} m")
    print(f"Calculated Shift Forward: {dy:+.4f} m")
    print(f"Accumulated Drift Right:  {vo.accumulated_x:+.4f} m")
    print(f"Accumulated Drift Fwd:    {vo.accumulated_y:+.4f} m")
    print("==================================\n")
