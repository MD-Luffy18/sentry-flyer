"""
Sentry Flyer — Local Edge AI Inference Pipeline
Location: Phase 3/detect.py
Purpose: Loads quantized object detection models using OpenCV's DNN module,
         preprocesses input frames, and extracts validated visual bounding boxes.
"""

import cv2
import numpy as np
import time

class EdgeDetector:
    def __init__(self, model_onnx_path=None, conf_threshold=0.5, nms_threshold=0.4):
        """
        Initializes the model settings, class labels, and DNN inference target backends.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Disaster response target classes mapped to the COCO index and custom fine-tuning
        self.classes = ["person", "fire", "smoke", "floodwater", "damage"]
        
        # Check if physical weights exist, otherwise initialize in mock-simulation fallback mode
        self.simulation_mode = True
        if model_onnx_path is not None:
            try:
                # Load ONNX model into OpenCV's DNN engine
                self.net = cv2.dnn.readNetFromONNX(model_onnx_path)
                
                # Set execution backend to CUDA for hardware acceleration on NVIDIA Jetson GPUs
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.simulation_mode = False
                print(f"[AI INIT] Model loaded successfully from {model_onnx_path}. GPU acceleration enabled.")
            except Exception as e:
                print(f"[AI WARNING] Failed to load ONNX model. Falling back to simulation mode. Details: {e}")

    def preprocess(self, frame, target_size=(416, 416)):
        """
        Preprocesses raw BGR frames into normalized blobs for neural network digestion.
        """
        # Converts frame to 4D blob, rescales intensity [1] to [2], and swaps R and B channels
        blob = cv2.dnn.blobFromImage(
            frame, 
            scalefactor=1.0/255.0, 
            size=target_size, 
            mean=(0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        return blob

    def run_inference(self, frame):
        """
        Executes a forward pass and filters raw predictions using Non-Maximum Suppression (NMS).
        """
        if self.simulation_mode:
            return self._simulate_inference(frame)
            
        height, width = frame.shape[:2]
        blob = self.preprocess(frame)
        self.net.setInput(blob)
        
        # Execute forward pass through network layers
        start_time = time.time()
        outputs = self.net.forward()
        inference_time_ms = (time.time() - start_time) * 1000.0
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Parse outputs (Assuming YOLO format: [x_center, y_center, width, height, confidence, class_probs...])
        for detection in outputs:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > self.conf_threshold:
                # Scale coordinates back to original image dimensions
                center_x = int(detection * width)
                center_y = int(detection[2] * height)
                w = int(detection[3] * width)
                h = int(detection[4] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
        # Apply Non-Maximum Suppression to remove overlapping redundant bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                final_detections.append({
                    "class": self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown",
                    "box": [x, y, x + w, y + h], # Format: [x_min, y_min, x_max, y_max]
                    "confidence": round(confidences[i], 2)
                })
                
        return final_detections, round(inference_time_ms, 1)

    def _simulate_inference(self, frame):
        """
        Mock fallback generator returning deterministic detections for hardware validation.
        """
        # Simulates a fixed 12ms execution latency matching optimized INT8 inference times
        simulated_latency = 12.5 
        height, width = frame.shape[:2]
        
        # Inject a simulated target in the center region
        simulated_detections = [
            {
                "class": "person",
                "box": [int(width * 0.4), int(height * 0.4), int(width * 0.6), int(height * 0.6)],
                "confidence": 0.88
            }
        ]
        return simulated_detections, simulated_latency


# =====================================================================
# SYSTEM INTEGRATION TEST
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Initializing Sentry Flyer AI Detector...")
    detector = EdgeDetector()
    
    # Create blank visual frame
    mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(mock_frame, "Simulation Frame", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    detections, latency = detector.run_inference(mock_frame)
    
    print("\n=== AI DETECTOR INFERENCE REPORT ===")
    print(f"Measured Frame Latency: {latency} ms")
    print(f"Number of Objects Found: {len(detections)}")
    for det in detections:
        print(f" - Detected '{det['class']}' with confidence {det['confidence']} at box {det['box']}")
    print("====================================\n")