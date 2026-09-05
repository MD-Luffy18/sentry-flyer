"""
Sentry Flyer — Visual Odometry GPS-Denied Fallback Engine
Location: Phase 2/odometry.py
Purpose: Tracks high-contrast ground features across video frames using Lucas-Kanade 
         optical flow, estimating physical displacement when GPS signals fail.
"""

import cv2
import numpy as np
import time


class VisualOdometry:
    def __init__(self, max_features=100, scaling_factor=0.05):
        """
        Initializes the visual tracker with feature limits and calibration scales.
        """
        self.max_features = max_features
        self.scaling_factor = scaling_factor  # Translates pixel movements to estimated physical meters
        
        # Parameters for Shi-Tomasi corner detection (finding reliable tracking points)
        self.feature_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Parameters for Lucas-Kanade optical flow tracking
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # State tracking variables
        self.prev_gray = None
        self.prev_points = None
        self.accumulated_x = 0.0  # Estimated drift on X-axis (meters)
        self.accumulated_y = 0.0  # Estimated drift on Y-axis (meters)
        self.last_frame_time = None

    def initialize_tracking(self, frame):
        """
        Initializes tracking by converting the first frame to grayscale 
        and detecting highly trackable corners.
        """
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        self.last_frame_time = time.time()
        
        if self.prev_points is not None:
            print(f"[VO INIT] Successfully locked onto {len(self.prev_points)} distinct terrain features.")
        else:
            print("[VO WARNING] Low texture terrain! Failed to locate trackable features.")

    def update_odometry(self, current_frame, current_altitude):
        """
        Calculates frame-to-frame feature motion and scales it using drone altitude.
        
        Parameters:
            current_frame (np.ndarray): The latest camera frame (BGR format)
            current_altitude (float): Live relative altitude from barometer (meters)
            
        Returns:
            delta_x (float): Shift in meters on X-axis since last update
            delta_y (float): Shift in meters on Y-axis since last update
        """
        if self.prev_gray is None or self.prev_points is None:
            self.initialize_tracking(current_frame)
            return 0.0, 0.0

        # Convert incoming frame to grayscale for faster processing
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()
        dt = current_time - self.last_frame_time
        
        if dt <= 0.0:
            return 0.0, 0.0

        # Calculate Lucas-Kanade optical flow between consecutive frames
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        # Select only the features that were successfully tracked in both frames
        if next_points is not None and status is not None:
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]
        else:
            good_new = np.array([])
            good_old = np.array([])

        # Handle failure case: If we lose too many points, re-detect features and return
        if len(good_new) < 10:
            print("[VO ALERT] Tracking points depleted! Re-initializing feature grid...")
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.last_frame_time = current_time
            return 0.0, 0.0

        # Calculate average pixel displacement across all tracked features
        diffs = good_new - good_old
        avg_pixel_dx = np.mean(diffs[:, 0])
        avg_pixel_dy = np.mean(diffs[:, 1])

        # Altitude Scaling: A pixel shift at 10m covers less physical ground than at 50m.
        # We adjust our physical estimation dynamically using the barometer's altitude.
        altitude_scale = current_altitude * self.scaling_factor
        
        # Convert pixel motion to estimated physical meters
        delta_x = float(avg_pixel_dx * altitude_scale)
        delta_y = float(avg_pixel_dy * altitude_scale)

        # Accumulate estimated total drift distance from origin
        self.accumulated_x += delta_x
        self.accumulated_y += delta_y

        # Update state for the next frame transaction
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        self.last_frame_time = current_time

        # Every 10 frames, refresh the feature list to discard aged or out-of-frame corners
        if len(self.prev_points) < (self.max_features * 0.6):
            new_features = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            if new_features is not None:
                self.prev_points = np.vstack((self.prev_points, new_features))
                # Deduplicate coordinates and keep within limit
                self.prev_points = self.prev_points[:self.max_features]

        return delta_x, delta_y


# =====================================================================
# INTEGRATION TESTING RUN (Executable validation)
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Initializing Sentry Flyer Visual Odometry Tracker...")
    vo = VisualOdometry()

    # Generate mock camera frame 1 (Solid gray background with 4 distinct black tracking blocks)
    frame_t0 = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame_t0, (100, 100), (120, 120), (0, 0, 0), -1)
    cv2.rectangle(frame_t0, (500, 100), (520, 120), (0, 0, 0), -1)
    cv2.rectangle(frame_t0, (100, 350), (120, 370), (0, 0, 0), -1)
    cv2.rectangle(frame_t0, (500, 350), (520, 370), (0, 0, 0), -1)

    # Initialize tracking with frame 1
    vo.initialize_tracking(frame_t0)

    # Generate mock camera frame 2 (Simulate drone drifting right: all physical features move left by 8 pixels)
    frame_t1 = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame_t1, (92, 100), (112, 120), (0, 0, 0), -1)
    cv2.rectangle(frame_t1, (492, 100), (512, 120), (0, 0, 0), -1)
    cv2.rectangle(frame_t1, (92, 350), (112, 370), (0, 0, 0), -1)
    cv2.rectangle(frame_t1, (492, 350), (512, 370), (0, 0, 0), -1)

    # Calculate drift assuming a flight altitude of 20 meters
    dx, dy = vo.update_odometry(frame_t1, current_altitude=20.0)

    print("\n=== VISUAL ODOMETRY CALCULATOR ===")
    print(f"Calculated Shift X:  {dx:.4f} meters")
    print(f"Calculated Shift Y:  {dy:.4f} meters")
    print(f"Accumulated Drift X: {vo.accumulated_x:.4f} meters")
    print(f"Accumulated Drift Y: {vo.accumulated_y:.4f} meters")
    print("==================================\n")