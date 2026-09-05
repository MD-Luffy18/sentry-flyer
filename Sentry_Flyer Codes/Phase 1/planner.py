"""
Sentry Flyer — MAVLink Mission Planner & Geofence Controller
Location: Phase 1/planner.py
Purpose: Establishes a telemetry link with the Pixhawk flight controller,
         tracks live aircraft position, enforces a circular geofence, and
         generates lawnmower search patterns.

Safety model: this module never commands motors. The only override it can
issue is a mode change to RTL (return-to-launch), which the flight controller
executes with its own stabilisation loop.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

log = logging.getLogger("sentry.planner")

EARTH_RADIUS_M = 6_371_000.0
METERS_PER_DEG_LAT = 111_111.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    d_lat = rlat2 - rlat1
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(d_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def generate_lawnmower_waypoints(
    sw_lat: float,
    sw_lon: float,
    ne_lat: float,
    ne_lon: float,
    rows: Optional[int] = None,
    swath_m: Optional[float] = None,
) -> list[tuple[float, float]]:
    """
    Lawnmower (boustrophedon) sweep over the bounding box (sw, ne).

    Pass either `rows` (explicit number of east-west passes) or `swath_m`
    (the camera's ground footprint width; rows are derived so adjacent passes
    touch). Returns a list of (lat, lon) waypoints; each row contributes its
    two ends, alternating direction so the drone never backtracks.
    """
    if rows is None:
        if swath_m is None or swath_m <= 0:
            raise ValueError("Provide rows>=1 or swath_m>0")
        span_m = abs(ne_lat - sw_lat) * METERS_PER_DEG_LAT
        rows = max(1, math.ceil(span_m / swath_m) + 1)
    if rows < 1:
        raise ValueError("rows must be >= 1")

    lat_step = (ne_lat - sw_lat) / (rows - 1) if rows > 1 else 0.0
    waypoints: list[tuple[float, float]] = []
    for i in range(rows):
        lat = sw_lat + i * lat_step
        west, east = (lat, sw_lon), (lat, ne_lon)
        waypoints.extend((west, east) if i % 2 == 0 else (east, west))
    return waypoints


class SentryPlanner:
    def __init__(
        self,
        connection_string: str = "/dev/ttyTHS1",
        baud_rate: int = 921600,
        fence_center_lat: float = 25.31000,
        fence_center_lon: float = 78.48800,
        fence_radius_m: float = 500.0,
        heartbeat_timeout_s: float = 30.0,
    ):
        """
        Open the MAVLink link. Defaults target the Jetson Orin Nano's hardware
        serial port; use "udpin:localhost:14550" for a SITL simulator.
        """
        # Imported lazily so the pure-math helpers above are usable without pymavlink.
        from pymavlink import mavutil

        self._mavutil = mavutil
        log.info("Connecting to Pixhawk on %s at %d baud", connection_string, baud_rate)
        self.vehicle = mavutil.mavlink_connection(connection_string, baud=baud_rate)

        log.info("Waiting for vehicle heartbeat (timeout %.0fs)...", heartbeat_timeout_s)
        if self.vehicle.wait_heartbeat(timeout=heartbeat_timeout_s) is None:
            raise TimeoutError("No MAVLink heartbeat received")
        log.info(
            "Heartbeat received. System ID: %s, Component ID: %s",
            self.vehicle.target_system,
            self.vehicle.target_component,
        )

        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.last_fix_time: Optional[float] = None

        self.fence_center_lat = fence_center_lat
        self.fence_center_lon = fence_center_lon
        self.fence_radius_meters = fence_radius_m
        self.rtl_engaged = False

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    def request_data_stream(self, rate_hz: int = 10) -> None:
        """Ask the autopilot for GLOBAL_POSITION_INT at `rate_hz`."""
        mav = self._mavutil.mavlink
        # Modern autopilots honour SET_MESSAGE_INTERVAL (interval in microseconds).
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mav.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mav.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            int(1_000_000 / rate_hz),
            0, 0, 0, 0, 0,
        )
        # Legacy stream request kept for older ArduPilot builds.
        self.vehicle.mav.request_data_stream_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mav.MAV_DATA_STREAM_POSITION,
            rate_hz,
            1,
        )

    def read_telemetry(self) -> bool:
        """Drain one GLOBAL_POSITION_INT packet if available. Returns True on update."""
        msg = self.vehicle.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if not msg:
            return False
        # The autopilot scales lat/lon by 1e7 and altitude by 1e3 to send integers.
        self.current_lat = msg.lat / 1.0e7
        self.current_lon = msg.lon / 1.0e7
        self.current_alt = msg.relative_alt / 1000.0
        self.last_fix_time = time.time()
        return True

    @property
    def has_fix(self) -> bool:
        return self.last_fix_time is not None or (self.current_lat != 0.0 and self.current_lon != 0.0)

    # ------------------------------------------------------------------ #
    # Geofence
    # ------------------------------------------------------------------ #
    def calculate_distance_to_center(self) -> float:
        """Great-circle distance in metres from the fence centre to the aircraft."""
        return haversine_m(self.fence_center_lat, self.fence_center_lon, self.current_lat, self.current_lon)

    def check_geofence_breach(self) -> bool:
        """
        Compare live position to the fence and engage RTL once on breach.
        Returns True while the aircraft is outside the fence.
        """
        if not self.has_fix:
            return False

        distance = self.calculate_distance_to_center()
        if distance <= self.fence_radius_meters:
            return False

        if not self.rtl_engaged:
            log.critical(
                "GEOFENCE BREACHED: %.1f m from centre exceeds %.0f m limit",
                distance,
                self.fence_radius_meters,
            )
            self.trigger_return_to_home()
        return True

    def trigger_return_to_home(self) -> None:
        """Override the current mission with RTL. Idempotent."""
        if self.rtl_engaged:
            return
        log.critical("Engaging RETURN-TO-LAUNCH on flight controller")
        self.vehicle.set_mode("RTL")
        self.rtl_engaged = True

    # Kept as a method for backwards compatibility with earlier callers.
    def generate_lawnmower_waypoints(self, sw_lat, sw_lon, ne_lat, ne_lon, rows=5, swath_m=None):
        return generate_lawnmower_waypoints(sw_lat, sw_lon, ne_lat, ne_lon, rows=rows, swath_m=swath_m)


# =====================================================================
# HARDWARE TELEMETRY TEST EMULATION RUN
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("[TEST] Lawnmower pattern over a 100 m x 100 m box with a 30 m swath:")
    for wp in generate_lawnmower_waypoints(25.3100, 78.4880, 25.3109, 78.4890, swath_m=30.0):
        print(f"  ({wp[0]:.5f}, {wp[1]:.5f})")

    # Without a physical serial line, bind to the UDP loopback used by SITL simulators.
    print("\n[TEST] Running local emulation loop on UDP loopback (udpin:localhost:14550)...")
    try:
        planner = SentryPlanner(connection_string="udpin:localhost:14550", heartbeat_timeout_s=5.0)
        planner.request_data_stream()

        print("\n[TEST] Emulating active boundary checking...")
        for step in range(5):
            # Inject coordinates heading slowly out of bounds.
            planner.current_lat = 25.31000 + step * 0.0015
            planner.current_lon = 78.48800 + step * 0.0015
            planner.last_fix_time = time.time()

            dist = planner.calculate_distance_to_center()
            print(f"Step {step + 1} | Pos: ({planner.current_lat:.5f}, {planner.current_lon:.5f}) | Dist: {dist:.1f}m")
            if planner.check_geofence_breach():
                break
            time.sleep(0.5)
    except Exception as e:  # noqa: BLE001 - surface any link failure to the operator
        print(f"\n[EMULATOR STOPPED] Local device not connected. Details: {e}")
        print("[INFO] Run this on the Jetson connected to the Pixhawk's TELEM1/TELEM2 port.")
