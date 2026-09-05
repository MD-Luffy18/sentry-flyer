"""
Sentry Flyer — 3D Ground-Coordinate Projection Engine
Location: Phase 2/projection.py
Purpose: Translates 2D image pixel coordinates into real-world GPS coordinates
         (WGS-84 Latitude/Longitude) using drone altitude, orientation, and gimbal angles.
"""

import math

class CoordinateProjector:
    def __init__(self, rgb_w=1920, rgb_h=1080, h_fov=100.0):
        """
        Initializes the projector with camera sensor specifications.
        """
        self.image_width = rgb_w
        self.image_height = rgb_h
        self.h_fov = h_fov
        # Calculate vertical FOV based on sensor aspect ratio
        self.v_fov = h_fov * (rgb_h / rgb_w)

    def project_pixel_to_gps(self, drone_lat, drone_lon, drone_alt, drone_yaw, gimbal_pitch, pixel_x, pixel_y):
        """
        Calculates real-world GPS coordinates of a target in a camera frame.
        
        Parameters:
            drone_lat (float): Live Latitude of the drone (degrees)
            drone_lon (float): Live Longitude of the drone (degrees)
            drone_alt (float): Live Altitude of the drone (meters from ground)
            drone_yaw (float): Compass heading of the drone (degrees, 0-359)
            gimbal_pitch (float): Pitch tilt of the camera gimbal (degrees, e.g. -45.0)
            pixel_x (int): Horizontal pixel column index of target (0 to width)
            pixel_y (int): Vertical pixel row index of target (0 to height)
            
        Returns:
            target_lat (float): Projected target Latitude (degrees)
            target_lon (float): Projected target Longitude (degrees)
        """
        # Convert all angular system values from degrees to radians
        yaw_rad = math.radians(drone_yaw)
        pitch_rad = math.radians(gimbal_pitch)

        # Normalize pixel coordinates relative to the sensor frame center (-1.0 to 1.0)
        offset_x = (pixel_x - (self.image_width / 2.0)) / (self.image_width / 2.0)
        offset_y = ((self.image_height / 2.0) - pixel_y) / (self.image_height / 2.0)

        # Convert normalized offsets to precise angles relative to the optical axis
        pixel_angle_x = offset_x * (math.radians(self.h_fov) / 2.0)
        pixel_angle_y = offset_y * (math.radians(self.v_fov) / 2.0)

        # Compute net physical pitch angle of target relative to the horizontal plane
        total_pitch_rad = pitch_rad + pixel_angle_y

        # Safety Check: Limit calculation angles near or above the horizon to prevent division errors
        if total_pitch_rad >= -0.087:  # Under 5 degrees below horizon
            total_pitch_rad = -0.087

        # Use trigonometric projection to get the flat horizontal ground distance to target
        ground_distance = drone_alt / math.tan(-total_pitch_rad)

        # Calculate absolute bearing of the target line-of-sight vector
        target_bearing = yaw_rad + pixel_angle_x

        # Define Earth coordinate scale factors (1 deg Lat ≈ 111,111 meters)
        meters_per_degree_lat = 111111.0
        # Longitude scale factor shrinks as we move away from the Equator
        meters_per_degree_lon = 111111.0 * math.cos(math.radians(drone_lat))

        # Project coordinate deltas
        delta_lat = (ground_distance * math.cos(target_bearing)) / meters_per_degree_lat
        delta_lon = (ground_distance * math.sin(target_bearing)) / meters_per_degree_lon

        # Add deltas to aircraft coordinate origin
        target_lat = drone_lat + delta_lat
        target_lon = drone_lon + delta_lon

        return round(target_lat, 6), round(target_lon, 6)


# =====================================================================
# SYSTEM INTEGRATION TEST
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Initializing Sentry Flyer Coordinate Projector...")
    projector = CoordinateProjector()
    
    # Simulate: Drone at (25.31000, 78.48800) at 50m altitude, facing straight North (0.0 yaw).
    # Gimbal is looking down at a 45-degree angle (-45.0 pitch).
    # Target is detected directly in the center of the visual frame.
    lat, lon = projector.project_pixel_to_gps(
        drone_lat=25.31000,
        drone_lon=78.48800,
        drone_alt=50.0,
        drone_yaw=0.0,
        gimbal_pitch=-45.0,
        pixel_x=960,
        pixel_y=540
    )
    
    print("\n=== COORDINATE ESTIMATION RESULTS ===")
    print(f"Calculated Target GPS Lat: {lat:.6f}")
    print(f"Calculated Target GPS Lon: {lon:.6f}")
    print("=====================================\n")