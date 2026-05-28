from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "This script requires cv2 and numpy. Run `conda activate cv` first."
    ) from exc

from model_path_helper import native_safe_model_path

DETECTOR_HOG = "hog"
DETECTOR_MEDIAPIPE = "mediapipe"
DETECTOR_YUNET = "yunet"
DETECTORS = (DETECTOR_HOG, DETECTOR_MEDIAPIPE, DETECTOR_YUNET)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
MEDIAPIPE_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def area(self) -> int:
        return self.w * self.h


class PersonDetector:
    name = "base"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HOGPersonDetector(PersonDetector):
    name = DETECTOR_HOG

    def __init__(self, max_width: int = 960, min_height_ratio: float = 0.12) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.max_width = max_width
        self.min_height_ratio = min_height_ratio

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        scale = 1.0
        resized = frame
        if width > self.max_width:
            scale = self.max_width / width
            resized = cv2.resize(frame, None, fx=scale, fy=scale)

        rects, weights = self.hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.03,
        )

        detections: List[Detection] = []
        min_height = int(resized.shape[0] * self.min_height_ratio)
        for (x, y, w, h), score in zip(rects, weights):
            if h < min_height:
                continue
            inv_scale = 1.0 / scale
            detections.append(
                Detection(
                    x=int(x * inv_scale),
                    y=int(y * inv_scale),
                    w=int(w * inv_scale),
                    h=int(h * inv_scale),
                    score=float(score),
                )
            )
        return non_max_suppression(detections, iou_threshold=0.35)


class MediaPipeFaceDetector(PersonDetector):
    name = DETECTOR_MEDIAPIPE

    def __init__(
        self,
        model_path: Path = MEDIAPIPE_MODEL_PATH,
        min_confidence: float = 0.5,
    ) -> None:
        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe model not found: {model_path}\n"
                "Download it with:\n"
                "  curl -fsSL -o models/blaze_face_short_range.tflite \\\n"
                "    https://storage.googleapis.com/mediapipe-models/face_detector/"
                "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is not installed. Run `pip install mediapipe` "
                "inside the cv environment, or use --detector yunet."
            ) from exc

        safe_model_path = native_safe_model_path(model_path)
        self._mp = mp
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(safe_model_path)),
            min_detection_confidence=min_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        detections: List[Detection] = []
        for det in result.detections:
            box = det.bounding_box
            x = max(0, int(box.origin_x))
            y = max(0, int(box.origin_y))
            w = min(int(box.width), width - x)
            h = min(int(box.height), height - y)
            if w <= 0 or h <= 0:
                continue
            score = det.categories[0].score if det.categories else 0.0
            detections.append(Detection(x=x, y=y, w=w, h=h, score=float(score)))
        return detections

    def close(self) -> None:
        try:
            self.detector.close()
        except Exception:
            pass


class YuNetFaceDetector(PersonDetector):
    name = DETECTOR_YUNET

    def __init__(
        self,
        model_path: Path = YUNET_MODEL_PATH,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "cv2.FaceDetectorYN is unavailable. Upgrade opencv "
                "(opencv-python>=4.6) or use --detector mediapipe."
            )
        if not model_path.exists():
            raise RuntimeError(
                f"YuNet model not found: {model_path}\n"
                "Download it with:\n"
                "  curl -fsSL -o models/face_detection_yunet_2023mar.onnx \\\n"
                "    https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )
        safe_model_path = native_safe_model_path(model_path)
        self.detector = cv2.FaceDetectorYN.create(
            str(safe_model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            5000,
        )
        self._input_size: Tuple[int, int] = (0, 0)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        if self._input_size != (width, height):
            self.detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, faces = self.detector.detect(frame)
        detections: List[Detection] = []
        if faces is None:
            return detections
        for face in faces:
            x, y, w, h = (int(round(v)) for v in face[:4])
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            if w <= 0 or h <= 0:
                continue
            score = float(face[-1])
            detections.append(Detection(x=x, y=y, w=w, h=h, score=score))
        return detections


def intersection_over_union(a: Detection, b: Detection) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(
    detections: List[Detection],
    iou_threshold: float,
) -> List[Detection]:
    kept: List[Detection] = []
    for det in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(intersection_over_union(det, prev) > iou_threshold for prev in kept):
            continue
        kept.append(det)
    return kept


def make_detector(name: str) -> PersonDetector:
    if name == DETECTOR_HOG:
        return HOGPersonDetector()
    if name == DETECTOR_MEDIAPIPE:
        return MediaPipeFaceDetector()
    if name == DETECTOR_YUNET:
        return YuNetFaceDetector()
    raise RuntimeError(f"Unknown detector: {name}")
