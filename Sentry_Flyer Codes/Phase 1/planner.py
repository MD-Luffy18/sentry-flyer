"""
Sentry Flyer — MAVLink Mission Planner & Geofence Controller
Location: Phase 1/planner.py
Purpose: Establishes low-latency telemetry link with Pixhawk flight controller,
         tracks live aircraft position, and enforces autonomous geofence boundaries.
"""

import time
import math
from pymavlink import mavutil

class SentryPlanner:
    def __init__(self, connection_string="/dev/ttyTHS1", baud_rate=921600):
        """
        Initializes the MAVLink connection.
        Default parameters are set for the Jetson Orin Nano's hardware serial port.
        """
        print(f"[INIT] Connecting to Pixhawk on {connection_string} at {baud_rate} baud...")
        # Establish serial connection using MAVLink protocol
        self.vehicle = mavutil.mavlink_connection(connection_string, baud=baud_rate)
        
        # Wait for first heartbeat packet from Pixhawk to confirm connection
        print("[INIT] Waiting for vehicle system heartbeat...")
        self.vehicle.wait_heartbeat()
        print(f"[INIT] Heartbeat received! System ID: {self.vehicle.target_system}, Component ID: {self.vehicle.target_component}")
        
        # Cache variables for tracking drone position
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        
        # Geofence parameters (Center coordinate and absolute safety radius in meters)
        self.fence_center_lat = 25.31000
        self.fence_center_lon = 78.48800
        self.fence_radius_meters = 500.0  # Limit drone to a 500m operations bubble

    def request_data_stream(self):
        """
        Requests specific high-frequency telemetry data streams from the Pixhawk.
        """
        # Request global position data (lat/lon/alt) at 10 Hz
        self.vehicle.mav.request_data_stream_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            10,  # Rate in Hz
            1    # 1 to start stream, 0 to stop
        )

    def read_telemetry(self):
        """
        Reads incoming MAVLink packets and updates current aircraft coordinates.
        """
        # Read a non-blocking incoming MAVLink message
        msg = self.vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg:
            # Pixhawk scales raw GPS coordinate floats by 1E7 to transmit as integers
            self.current_lat = msg.lat / 1.0e7
            self.current_lon = msg.lon / 1.0e7
            self.current_alt = msg.relative_alt / 1000.0  # Convert millimeters to meters
            return True
        return False

    def calculate_distance_to_center(self):
        """
        Calculates the 2D planar distance from the geofence center to the drone.
        Uses Haversine spherical math to maintain sub-meter precision in the field.
        """
        earth_radius = 6371000.0  # Earth's radius in meters
        
        # Convert degrees to radians
        lat1 = math.radians(self.fence_center_lat)
        lon1 = math.radians(self.fence_center_lon)
        lat2 = math.radians(self.current_lat)
        lon2 = math.radians(self.current_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine calculation
        a = (math.sin(dlat / 2.0) ** 2) + \
            (math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2.0) ** 2))
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        
        return earth_radius * c

    def check_geofence_breach(self):
        """
        Monitors live location and triggers automatic failsafe if boundary is breached.
        """
        if self.current_lat == 0.0 or self.current_lon == 0.0:
            # Skip check if GPS data is not yet initialized
            return False
            
        distance = self.calculate_distance_to_center()
        
        if distance > self.fence_radius_meters:
            print(f"[FAILSAFE ALERT] GEOFENCE BREACHED! Distance: {distance:.2f}m exceeds limit of {self.fence_radius_meters}m.")
            self.trigger_return_to_home()
            return True
        return False

    def trigger_return_to_home(self):
        """
        Sends a high-priority command to Pixhawk overriding current mission to RTL.
        """
        print("[FAILSAFE COMMAND] Force-engaging RETURN-TO-LAUNCH (RTL) mode on Pixhawk...")
        
        # MAVLink custom mode for RTL in ArduPilot is mode number 6
        # We target the flight controller using system ID and component ID
        self.vehicle.set_mode('RTL')

    def generate_lawnmower_waypoints(self, sw_lat, sw_lon, ne_lat, ne_lon, rows=5):
        """
        Algorithmic coordinates sweep generator (Lawnmower sweep pattern).
        Generates snake-like grid pathing bounds for systematic area search.
        """
        grid_waypoints = []
        lat_step = (ne_lat - sw_lat) / (rows - 1)
        
        for i in range(rows):
            # Calculate target row latitude
            target_lat = sw_lat + (i * lat_step)
            
            # Snake-like sweep direction toggle
            if i % 2 == 0:
                # West to East
                grid_waypoints.append((target_lat, sw_lon))
                grid_waypoints.append((target_lat, ne_lon))
            else:
                # East to West
                grid_waypoints.append((target_lat, ne_lon))
                grid_waypoints.append((target_lat, sw_lon))
                
        return grid_waypoints


# =====================================================================
# HARDWARE TELEMENTRY TEST EMULATION RUN
# =====================================================================
if __name__ == "__main__":
    # For local system software dry-runs without a physical hardware serial line,
    # we bind the telemetry port to a local UDP loopback stream (port 14550) used by simulators.
    print("[TEST] Running Local Emulation Loop on standard UDP loopback...")
    try:
        planner = SentryPlanner(connection_string="udpin:localhost:14550")
        planner.request_data_stream()
        
        print("\n[TEST] Emulating active boundary checking...")
        # Simulate active coordinate ingestion
        for step in range(5):
            # Inject simulated coordinates heading slowly out of bounds
            planner.current_lat = 25.31000 + (step * 0.0015)
            planner.current_lon = 78.48800 + (step * 0.0015)
            
            dist = planner.calculate_distance_to_center()
            print(f"Step {step+1} | Pos: ({planner.current_lat:.5f}, {planner.current_lon:.5f}) | Dist: {dist:.1f}m")
            
            # Check boundary limits
            breached = planner.check_geofence_breach()
            if breached:
                break
            time.sleep(0.5)
            
    except Exception as e:
        print(f"\n[EMULATOR STOPPED] Local device not connected. Details: {e}")
        print("[INFO] Use this script on the physical Jetson connected to the Pixhawk's TELEM1/TELEM2 telemetry port.")