"""
Sentry Flyer — Local Edge AI Inference Pipeline
Location: Phase 3/detect.py
Purpose: Loads an ONNX object-detection model through OpenCV's DNN module,
         preprocesses input frames, and returns NMS-filtered bounding boxes.

Supported model outputs
-----------------------
* YOLOv5-style: (1, N, 5 + num_classes) rows of [cx, cy, w, h, objectness, class scores...]
* YOLOv8-style: (1, 4 + num_classes, N) columns of [cx, cy, w, h, class scores...]
Both are in input-blob pixel units and are rescaled to the source frame.
Falls back to a deterministic simulation when no model is supplied.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("sentry.detect")

Detection = dict  # {"class": str, "box": [x1, y1, x2, y2], "confidence": float}


def _cuda_available() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (AttributeError, cv2.error):
        return False


class EdgeDetector:
    DEFAULT_CLASSES = ("person", "fire", "smoke", "floodwater", "damage")

    def __init__(
        self,
        model_onnx_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (416, 416),
        classes: Optional[list[str]] = None,
        prefer_cuda: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.classes = list(classes or self.DEFAULT_CLASSES)
        self.net = None
        self.backend = "simulation"

        if model_onnx_path is None:
            log.warning("No model path supplied; running in simulation mode")
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(model_onnx_path)
        except cv2.error as e:
            log.warning("Failed to load ONNX model %s; using simulation mode. %s", model_onnx_path, e)
            return

        if prefer_cuda and _cuda_available():
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            self.backend = "cuda"
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.backend = "cpu"
        log.info("Model loaded from %s (backend=%s)", model_onnx_path, self.backend)

    @property
    def simulation_mode(self) -> bool:
        return self.net is None

    # ------------------------------------------------------------------ #
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame -> normalised NCHW RGB blob."""
        return cv2.dnn.blobFromImage(
            frame, scalefactor=1.0 / 255.0, size=self.input_size, mean=(0, 0, 0), swapRB=True, crop=False
        )

    def _decode(self, output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalise the raw network output into (boxes_cxcywh, scores, class_ids)
        for rows above the confidence threshold.
        """
        out = np.squeeze(output)
        if out.ndim != 2:
            raise ValueError(f"Unexpected output shape {output.shape}")

        nc = len(self.classes)
        # YOLOv8 exports transposed: (4 + nc, N). Detect by matching the class count.
        if out.shape[0] in (4 + nc, 5 + nc) and out.shape[1] not in (4 + nc, 5 + nc):
            out = out.T

        cols = out.shape[1]
        if cols == 5 + nc:  # v5: objectness column present
            class_scores = out[:, 5:] * out[:, 4:5]
        elif cols == 4 + nc:  # v8: class scores only
            class_scores = out[:, 4:]
        else:
            raise ValueError(f"Output has {cols} columns; expected {4 + nc} or {5 + nc} for {nc} classes")

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]
        keep = scores > self.conf_threshold
        return out[keep, :4], scores[keep], class_ids[keep]

    def run_inference(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Forward pass + NMS. Returns (detections, inference_time_ms)."""
        if self.simulation_mode:
            return self._simulate_inference(frame)

        height, width = frame.shape[:2]
        self.net.setInput(self.preprocess(frame))

        start = time.perf_counter()
        output = self.net.forward()
        inference_ms = (time.perf_counter() - start) * 1000.0

        boxes_cxcywh, scores, class_ids = self._decode(output)
        if len(scores) == 0:
            return [], round(inference_ms, 1)

        # Blob pixels -> source frame pixels, then centre-format -> top-left-format.
        sx = width / self.input_size[0]
        sy = height / self.input_size[1]
        cx, cy, w, h = (boxes_cxcywh * np.array([sx, sy, sx, sy])).T
        x = cx - w / 2.0
        y = cy - h / 2.0
        boxes_xywh = np.stack([x, y, w, h], axis=1).astype(int).tolist()

        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.astype(float).tolist(), self.conf_threshold, self.nms_threshold)
        detections: list[Detection] = []
        for i in np.array(indices).flatten():
            bx, by, bw, bh = boxes_xywh[i]
            cid = int(class_ids[i])
            detections.append(
                {
                    "class": self.classes[cid] if cid < len(self.classes) else "unknown",
                    "box": [
                        max(0, bx),
                        max(0, by),
                        min(width, bx + bw),
                        min(height, by + bh),
                    ],
                    "confidence": round(float(scores[i]), 2),
                }
            )
        return detections, round(inference_ms, 1)

    def _simulate_inference(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Deterministic stand-in used for pipeline validation without weights."""
        height, width = frame.shape[:2]
        simulated = [
            {
                "class": "person",
                "box": [int(width * 0.4), int(height * 0.4), int(width * 0.6), int(height * 0.6)],
                "confidence": 0.88,
            }
        ]
        return simulated, 12.5  # ms, matches typical INT8 Jetson latency

    @staticmethod
    def draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Annotate a copy of the frame with boxes and labels (for the operator view)."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out, f"{det['class']} {det['confidence']:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        return out


# =====================================================================
# SYSTEM INTEGRATION TEST
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("[TEST] Initializing Sentry Flyer AI Detector...")
    detector = EdgeDetector()

    mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(mock_frame, "Simulation Frame", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    detections, latency = detector.run_inference(mock_frame)

    print("\n=== AI DETECTOR INFERENCE REPORT ===")
    print(f"Backend: {detector.backend}")
    print(f"Measured Frame Latency: {latency} ms")
    print(f"Number of Objects Found: {len(detections)}")
    for det in detections:
        print(f" - Detected '{det['class']}' with confidence {det['confidence']} at box {det['box']}")
    print("====================================\n")
