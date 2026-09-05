"""
Sentry Flyer — 3D Ground-Coordinate Projection Engine
Location: Phase 2/projection.py
Purpose: Translates 2D image pixel coordinates into real-world GPS coordinates
         (WGS-84 latitude/longitude) using drone altitude, heading, and gimbal
         pitch, assuming locally flat ground.

Geometry
--------
A pinhole camera: a pixel offset from the optical centre corresponds to an
angle atan(offset_normalised * tan(fov / 2)), *not* a linear fraction of the
field of view. Vertical FOV is derived from horizontal FOV via the aspect
ratio of the tangents, which is the correct relation for a rectilinear lens.
"""

from __future__ import annotations

import math
from typing import Optional

METERS_PER_DEG_LAT = 111_111.0


class CoordinateProjector:
    def __init__(
        self,
        rgb_w: int = 1920,
        rgb_h: int = 1080,
        h_fov: float = 100.0,
        min_depression_deg: float = 5.0,
    ):
        """
        rgb_w, rgb_h: sensor resolution in pixels.
        h_fov: horizontal field of view in degrees.
        min_depression_deg: rays closer to the horizon than this are clamped;
            the projection is unreliable there and callers should treat the
            result as a rough bearing only.
        """
        self.image_width = rgb_w
        self.image_height = rgb_h
        self.h_fov = h_fov
        self.tan_half_h = math.tan(math.radians(h_fov) / 2.0)
        self.tan_half_v = self.tan_half_h * (rgb_h / rgb_w)
        self.v_fov = math.degrees(2.0 * math.atan(self.tan_half_v))
        self.min_depression_rad = math.radians(min_depression_deg)

    def pixel_angles(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """
        Angular offset (yaw_offset_rad, pitch_offset_rad) of a pixel from the
        optical axis. Positive yaw offset is to the right; positive pitch
        offset is up (towards the top of the image).
        """
        nx = (pixel_x - self.image_width / 2.0) / (self.image_width / 2.0)
        ny = (self.image_height / 2.0 - pixel_y) / (self.image_height / 2.0)
        return math.atan(nx * self.tan_half_h), math.atan(ny * self.tan_half_v)

    def project_pixel_to_gps(
        self,
        drone_lat: float,
        drone_lon: float,
        drone_alt: float,
        drone_yaw: float,
        gimbal_pitch: float,
        pixel_x: float,
        pixel_y: float,
    ) -> tuple[float, float]:
        """
        Ground coordinates of the pixel's line of sight.

        drone_alt: height above ground (m). drone_yaw: compass heading (deg).
        gimbal_pitch: camera pitch relative to horizontal (deg; -90 is nadir).
        Returns (lat, lon) rounded to 1e-6 degrees (~0.1 m).
        """
        if drone_alt <= 0.0:
            raise ValueError("drone_alt must be positive (height above ground)")

        yaw_offset, pitch_offset = self.pixel_angles(pixel_x, pixel_y)
        total_pitch = math.radians(gimbal_pitch) + pitch_offset

        # Clamp rays at or above the horizon so tan() cannot blow up.
        depression = max(-total_pitch, self.min_depression_rad)
        ground_distance = drone_alt / math.tan(depression)

        bearing = math.radians(drone_yaw) + yaw_offset
        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(drone_lat))

        target_lat = drone_lat + (ground_distance * math.cos(bearing)) / METERS_PER_DEG_LAT
        target_lon = drone_lon + (ground_distance * math.sin(bearing)) / meters_per_deg_lon
        return round(target_lat, 6), round(target_lon, 6)

    def ground_footprint_width(self, drone_alt: float, gimbal_pitch: float = -90.0) -> Optional[float]:
        """Width in metres of the ground strip seen at the image centre row (nadir by default)."""
        depression = -math.radians(gimbal_pitch)
        if depression <= 0:
            return None
        slant = drone_alt / math.sin(depression)
        return 2.0 * slant * self.tan_half_h


# =====================================================================
# SYSTEM INTEGRATION TEST
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Initializing Sentry Flyer Coordinate Projector...")
    projector = CoordinateProjector()
    print(f"Derived vertical FOV: {projector.v_fov:.2f} deg")
    print(f"Nadir footprint width at 50 m: {projector.ground_footprint_width(50.0):.1f} m")

    # Drone at (25.31, 78.488), 50 m up, heading north, gimbal 45 deg down,
    # target in the image centre -> 50 m due north.
    lat, lon = projector.project_pixel_to_gps(
        drone_lat=25.31000,
        drone_lon=78.48800,
        drone_alt=50.0,
        drone_yaw=0.0,
        gimbal_pitch=-45.0,
        pixel_x=960,
        pixel_y=540,
    )

    print("\n=== COORDINATE ESTIMATION RESULTS ===")
    print(f"Calculated Target GPS Lat: {lat:.6f}")
    print(f"Calculated Target GPS Lon: {lon:.6f}")
    print("=====================================\n")
