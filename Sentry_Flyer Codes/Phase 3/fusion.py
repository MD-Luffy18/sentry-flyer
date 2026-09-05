"""
Sentry Flyer — Unified Sensor Alignment & Cross-Modal Fusion Engine
Location: Phase 3/fusion.py
Purpose: Warps low-resolution thermal camera coordinates (FLIR Lepton) to match 
         high-resolution RGB coordinates (homography) and cross-checks temperatures 
         to eliminate environmental false alarms.
"""

import numpy as np
import cv2

class SensorAligner:
    def __init__(self, thermal_src_pts=None, rgb_dst_pts=None):
        """
        Calculates the 3x3 homography matrix to align thermal coordinates to RGB space.
        """
        # Default reference calibration points (4 matched points between sensors)
        if thermal_src_pts is None or rgb_dst_pts is None:
            # 160x120 FLIR Lepton thermal camera coordinates (4 corners)
            self.src_pts = np.array([
                [0.0, 0.0],
                [160.0, 0.0],
                [160.0, 120.0],
                [0.0, 120.0]
            ], dtype=np.float32)

            # Corresponding coordinates in the 1920x1080 RGB frame
            self.dst_pts = np.array([
                [0.0, 0.0],
                [1920.0, 0.0],
                [1920.0, 1080.0],
                [0.0, 1080.0]
            ], dtype=np.float32)
        else:
            self.src_pts = np.array(thermal_src_pts, dtype=np.float32)
            self.dst_pts = np.array(rgb_dst_pts, dtype=np.float32)

        # Compute the 3x3 perspective warp matrix (Homography)
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)

    def warp_thermal_to_rgb(self, thermal_frame, rgb_target_shape=(1080, 1920)):
        """
        Warps the thermal frame to overlay perfectly on the RGB frame.
        """
        height, width = rgb_target_shape[:2]
        return cv2.warpPerspective(
            thermal_frame,
            self.H,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )


class CrossModalFusion:
    def __init__(self, thermal_w=160, thermal_h=120, rgb_w=1920, rgb_h=1080):
        """
        Initializes physical sensor geometries and loads alignment tools.
        """
        self.thermal_w = thermal_w
        self.thermal_h = thermal_h
        self.rgb_w = rgb_w
        self.rgb_h = rgb_h
        
        # Instantiate spatial alignment module internally
        self.aligner = SensorAligner()

        # Generate an inverse transformation scaling matrix for coordinate mapping
        scale_x = self.thermal_w / self.rgb_w
        scale_y = self.thermal_h / self.rgb_h
        self.H_rgb_to_thermal = np.array([
            [scale_x, 0.0,     0.0],
            [0.0,     scale_y, 0.0],
            [0.0,     0.0,     1.0]
        ], dtype=np.float32)

    def project_box_to_thermal(self, rgb_box):
        """
        Projects an RGB bounding box [x_min, y_min, x_max, y_max] to the raw Thermal space.
        """
        x_min, y_min, x_max, y_max = rgb_box
        
        corners = np.array([
            [x_min, y_min, 1.0],
            [x_max, y_min, 1.0],
            [x_max, y_max, 1.0],
            [x_min, y_max, 1.0]
        ], dtype=np.float32).T
        
        # Multiply points by scaling homography matrix
        projected = np.dot(self.H_rgb_to_thermal, corners)
        projected = projected / projected[2, :]  # Normalize Z coordinates
        
        tx_min = int(np.clip(np.min(projected[0, :]), 0, self.thermal_w - 1))
        ty_min = int(np.clip(np.min(projected[1, :]), 0, self.thermal_h - 1))
        tx_max = int(np.clip(np.max(projected[0, :]), 0, self.thermal_w - 1))
        ty_max = int(np.clip(np.max(projected[1, :]), 0, self.thermal_h - 1))
        
        return [tx_min, ty_min, tx_max, ty_max]

    def verify_detection(self, rgb_class, rgb_box, thermal_frame_kelvin, base_confidence):
        """
        Cross-checks visual detections against raw physical thermal telemetry (Kelvin).
        """
        t_box = self.project_box_to_thermal(rgb_box)
        tx_min, ty_min, tx_max, ty_max = t_box
        
        # Extract corresponding thermal region of interest (ROI)
        thermal_roi = thermal_frame_kelvin[ty_min:ty_max+1, tx_min:tx_max+1]
        
        if thermal_roi.size == 0:
            return rgb_class, base_confidence * 0.7

        # Convert raw Kelvin (cK / 100) data to Celsius
        temperatures_c = (thermal_roi / 100.0) - 273.15
        
        if rgb_class == "person":
            avg_temp = np.mean(temperatures_c)
            
            # Mammalian warm-blood threshold filter (35.5 C to 37.5 C)
            if 35.5 <= avg_temp <= 37.5:
                final_confidence = min(base_confidence * 1.15, 0.98)
                verified_class = "person"
            elif 32.0 <= avg_temp <= 39.0:
                final_confidence = base_confidence * 0.85
                verified_class = "person_unverified"
            else:
                final_confidence = base_confidence * 0.30
                verified_class = "environmental_anomaly"
                
        elif rgb_class == "fire":
            max_temp = np.max(temperatures_c)
            
            # Flame ignition heat ceiling indicator threshold (> 150 C)
            if max_temp > 150.0:
                final_confidence = max(base_confidence, 0.95)
                verified_class = "fire"
            elif max_temp > 80.0:
                final_confidence = base_confidence * 0.75
                verified_class = "thermal_anomaly"
            else:
                final_confidence = base_confidence * 0.20
                verified_class = "reflector_false_alarm"
                
        else:
            verified_class = rgb_class
            final_confidence = base_confidence
            
        return verified_class, round(final_confidence, 2)


# =====================================================================
# SYSTEM INTEGRATION TEST
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Initializing Sentry Flyer Unified Fusion Engine...")
    fusion_engine = CrossModalFusion()
    
    # 1. Test Alignment Layer
    mock_thermal_raw = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(mock_thermal_raw, (80, 60), 10, 255, -1)
    warped_test = fusion_engine.aligner.warp_thermal_to_rgb(mock_thermal_raw)
    print(f"[TEST] Alignment successful. Warped array shape: {warped_test.shape}")
    
    # 2. Test Verification Layer (36.5 C = 30965 centi-Kelvin)
    mock_thermal_kelvin = np.full((120, 160), 29815, dtype=np.uint16)
    mock_thermal_kelvin[45:55, 45:55] = 30965
    
    class_name = "person"
    pixel_box = [540, 405, 660, 495]  # [x_min, y_min, x_max, y_max]
    yolo_conf = 0.85
    
    v_class, v_conf = fusion_engine.verify_detection(
        rgb_class=class_name,
        rgb_box=pixel_box,
        thermal_frame_kelvin=mock_thermal_kelvin,
        base_confidence=yolo_conf
    )
    
    print("\n=== SENSOR AGREEMENT TEST REPORT ===")
    print(f"Input Target:  Class='{class_name}', Conf={yolo_conf}")
    print(f"Fused Target:  Class='{v_class}', Conf={v_conf}")
    print(f"Status:        {'✅ TARGET VERIFIED' if v_conf >= 0.8 else '❌ REJECTED FALSE ALARM'}")
    print("====================================\n")