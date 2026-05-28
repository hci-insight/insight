from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

from camera_guidance import (
    CenterAlignment,
    compute_center_alignment,
    draw_center_alignment_overlay,
)
from model_path_helper import native_safe_model_path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "This script requires cv2 and numpy. Run `conda activate cv` first."
    ) from exc

try:
    import mss
except ImportError:
    mss = None


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "captures_compare"
MODELS_DIR = ROOT / "models"
WINDOW_NAME = "HCI Shot Capture Experiment"

MODE_MANUAL = "manual"
MODE_PERSON1 = "person1"
MODE_COUNT3 = "count3"
MODE_RATIO = "ratio"
MODES = (MODE_MANUAL, MODE_PERSON1, MODE_COUNT3, MODE_RATIO)

# Person/face detector backends.
# hog       : legacy OpenCV full-body HOG (window-scan, weak on top-down aerial shots)
# mediapipe : Google MediaPipe Face Detection (single-pass, all faces at once)
# yunet     : OpenCV YuNet ONNX face detector (single-pass, no heavy install)
DETECTOR_HOG = "hog"
DETECTOR_MEDIAPIPE = "mediapipe"
DETECTOR_YUNET = "yunet"
DETECTORS = (DETECTOR_HOG, DETECTOR_MEDIAPIPE, DETECTOR_YUNET)

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


@dataclass
class CaptureRecord:
    capture_id: str
    mode: str
    trigger: str
    timestamp: float
    elapsed_sec: float
    person_count: int
    people_area_ratio: float
    stable_hits: int
    raw_path: str
    overlay_path: str


# 얼굴 박스와 중심점 좌표를 코드 전체에서 같은 형태로 쓰기 위한 별칭입니다.
# BBox는 OpenCV 관례대로 (x, y, width, height)를 의미합니다.
BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


class DetectionLike(Protocol):
    """기존 Detection dataclass처럼 x, y, w, h를 가진 객체를 받기 위한 타입입니다."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class TrackedPerson:
    """Centroid Tracking으로 유지되는 한 사람의 현재 상태입니다."""

    object_id: int
    bbox: BBox
    centroid: Point
    disappeared: int = 0


@dataclass
class OverlapEvent:
    """얼굴 겹침이 감지되었을 때 안내에 필요한 정보를 담습니다."""

    object_id: int
    other_id: int
    clock_position: str
    message: str
    overlap_ratio: float


class FrameSource:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoSource(FrameSource):
    def __init__(self, source: Union[int, str]) -> None:
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self) -> None:
        self.capture.release()


class ScreenSource(FrameSource):
    def __init__(self, region: Tuple[int, int, int, int]) -> None:
        if mss is None:
            raise RuntimeError(
                "mss is not installed. Install it or use webcam/video source instead."
            )
        self.left, self.top, self.width, self.height = region
        self.sct = mss.mss()

    def read(self) -> Optional[np.ndarray]:
        shot = self.sct.grab(
            {
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height,
            }
        )
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self.sct.close()


class PersonDetector:
    """Common interface so the capture loop is detector-agnostic.

    Every backend returns axis-aligned boxes in the analysis-frame coordinate
    system. For face detectors the box is the face; for HOG it is the body.
    """

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
    """Google MediaPipe Face Detection (Tasks API, BlazeFace).

    A single forward pass over the whole frame returns every face at once,
    which is what the "한번에(one-shot) detection" feedback asks for. Works
    well on top-down aerial group shots where faces are clearer than bodies.
    """

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
        except ImportError as exc:  # pragma: no cover - environment guard
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
    """OpenCV YuNet ONNX face detector.

    Same one-shot, whole-frame behaviour as MediaPipe but needs no extra pip
    install beyond the small .onnx model file, so it is the fallback backend.
    """

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


def _as_bbox(detection: DetectionLike | Sequence[int]) -> BBox:
    """Detection 객체나 (x, y, w, h) 시퀀스를 공통 BBox 형태로 변환합니다."""

    if hasattr(detection, "x"):
        return (
            int(getattr(detection, "x")),
            int(getattr(detection, "y")),
            int(getattr(detection, "w")),
            int(getattr(detection, "h")),
        )
    if len(detection) < 4:
        raise ValueError("Detection sequence must contain x, y, w, h.")
    x, y, w, h = detection[:4]
    return int(x), int(y), int(w), int(h)


def bbox_centroid(bbox: BBox) -> Point:
    """박스 중심점을 계산합니다. Centroid Tracking의 기준점으로 사용됩니다."""

    x, y, w, h = bbox
    return int(x + w / 2), int(y + h / 2)


def bbox_area(bbox: BBox) -> int:
    """박스 면적을 계산합니다. 음수 크기가 들어와도 0 이상이 되게 막습니다."""

    return max(0, bbox[2]) * max(0, bbox[3])


def overlap_ratio(a: BBox, b: BBox) -> float:
    """두 얼굴 박스가 얼마나 겹쳤는지 계산합니다.

    일반 IoU가 아니라 "교집합 / 더 작은 박스 면적"을 씁니다. 얼굴 겹침에서는
    작은 얼굴 일부가 가려지는 상황도 중요해서 이 방식이 더 민감합니다.
    """

    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    # 두 박스가 겹치는 사각형의 좌표를 구합니다.
    inter_left = max(ax, bx)
    inter_top = max(ay, by)
    inter_right = min(ax + aw, bx + bw)
    inter_bottom = min(ay + ah, by + bh)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    smaller_area = max(1, min(bbox_area(a), bbox_area(b)))
    return inter_area / smaller_area


def clock_position(centroid: Point, frame_shape: Tuple[int, ...], center_radius_ratio: float = 0.20) -> str:
    """프레임 중심 기준으로 사람 위치를 시계 방향 표현으로 바꿉니다.

    예: 위쪽은 12시, 오른쪽은 3시, 아래쪽은 6시입니다.
    """

    frame_h, frame_w = frame_shape[:2]
    cx, cy = centroid
    dx = cx - frame_w / 2.0
    dy = cy - frame_h / 2.0

    # 중심 근처에 있으면 애매한 시계 방향 대신 "가운데"라고 말합니다.
    # 기본 반경을 넓게 잡아 중앙에 있는 사람이 더 안정적으로 안내되게 합니다.
    center_radius = min(frame_w, frame_h) * center_radius_ratio
    if math.hypot(dx, dy) <= center_radius:
        return "가운데"

    # atan2(dx, -dy)를 쓰면 12시 방향이 0도, 3시 방향이 90도가 됩니다.
    angle = math.degrees(math.atan2(dx, -dy))
    if angle < 0:
        angle += 360.0
    hour = int(round(angle / 30.0)) % 12
    hour = 12 if hour == 0 else hour
    return f"{hour}시"


class CentroidTracker:
    """프레임이 바뀌어도 같은 사람에게 같은 ID를 유지하는 간단한 추적기입니다."""

    def __init__(self, max_disappeared: int = 20, max_distance: float = 80.0) -> None:
        # max_disappeared: 검출이 끊겨도 몇 프레임까지 같은 사람으로 보관할지 정합니다.
        self.max_disappeared = max_disappeared
        # max_distance: 이전 중심점과 새 중심점이 이 거리보다 멀면 다른 사람으로 봅니다.
        self.max_distance = max_distance
        self.next_object_id = 1
        self.objects: Dict[int, TrackedPerson] = {}

    def update(self, boxes: Iterable[BBox]) -> Dict[int, TrackedPerson]:
        """새 프레임의 얼굴 박스 목록을 받아 추적 ID를 갱신합니다."""

        input_boxes = list(boxes)
        input_centroids = [bbox_centroid(box) for box in input_boxes]

        # 이번 프레임에 얼굴이 없으면 기존 객체의 disappeared 카운트만 올립니다.
        if not input_boxes:
            for object_id in list(self.objects):
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    del self.objects[object_id]
            return dict(self.objects)

        # 처음 들어온 얼굴들은 모두 새 사람으로 등록합니다.
        if not self.objects:
            for box, centroid in zip(input_boxes, input_centroids):
                self._register(box, centroid)
            return dict(self.objects)

        # 기존 객체 중심점과 새 검출 중심점 사이의 거리 행렬을 만듭니다.
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[object_id].centroid for object_id in object_ids]
        distances = self._distance_matrix(object_centroids, input_centroids)

        # 가장 가까운 쌍부터 매칭해서 ID가 튀는 것을 줄입니다.
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue

            # 기존 ID에 새 박스와 새 중심점을 연결합니다.
            object_id = object_ids[row]
            self.objects[object_id] = TrackedPerson(
                object_id=object_id,
                bbox=input_boxes[col],
                centroid=input_centroids[col],
                disappeared=0,
            )
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        unused_cols = set(range(len(input_boxes))) - used_cols

        # 매칭되지 않은 기존 객체는 잠시 사라진 것으로 처리합니다.
        for row in unused_rows:
            object_id = object_ids[row]
            self.objects[object_id].disappeared += 1
            if self.objects[object_id].disappeared > self.max_disappeared:
                del self.objects[object_id]

        # 매칭되지 않은 새 얼굴은 새 사람으로 등록합니다.
        for col in unused_cols:
            self._register(input_boxes[col], input_centroids[col])

        return dict(self.objects)

    def _register(self, bbox: BBox, centroid: Point) -> None:
        """새 object_id를 부여해 사람을 등록합니다."""

        object_id = self.next_object_id
        self.objects[object_id] = TrackedPerson(object_id, bbox, centroid)
        self.next_object_id += 1

    @staticmethod
    def _distance_matrix(a: List[Point], b: List[Point]) -> np.ndarray:
        """두 중심점 목록 사이의 유클리드 거리 행렬을 계산합니다."""

        matrix = np.zeros((len(a), len(b)), dtype="float32")
        for row, (ax, ay) in enumerate(a):
            for col, (bx, by) in enumerate(b):
                matrix[row, col] = math.hypot(ax - bx, ay - by)
        return matrix


class SpeechNotifier:
    """로컬 TTS가 가능하면 한국어 안내 문장을 음성으로 읽어주는 클래스입니다."""

    def __init__(self, enabled: bool = True, cooldown_sec: float = 2.0) -> None:
        self.enabled = enabled
        # 같은 안내가 너무 자주 반복되지 않도록 최소 간격을 둡니다.
        self.cooldown_sec = cooldown_sec
        self.last_spoken_at = 0.0
        self.last_message = ""
        self._pyttsx3 = None
        self._engine = None

        if not enabled:
            return
        try:
            # 설치되어 있으면 Python TTS 엔진을 우선 사용합니다.
            import pyttsx3  # type: ignore

            self._pyttsx3 = pyttsx3
            self._engine = pyttsx3.init()
            self._select_korean_voice()
        except Exception:
            self._pyttsx3 = None
            self._engine = None

    def speak(self, message: str) -> bool:
        """쿨다운 조건을 만족할 때만 안내 문장을 음성 출력합니다."""

        if not self.enabled:
            return False
        now = time.time()

        # 같은 문장 반복과 지나치게 빠른 연속 안내를 모두 막습니다.
        if message == self.last_message and (now - self.last_spoken_at) < self.cooldown_sec:
            return False
        if (now - self.last_spoken_at) < self.cooldown_sec:
            return False

        spoken = self._speak_now(message)
        if spoken:
            self.last_message = message
            self.last_spoken_at = now
        return spoken

    def _speak_now(self, message: str) -> bool:
        """사용 가능한 TTS 백엔드를 찾아 실제로 음성을 출력합니다."""

        # gTTS는 네트워크가 필요하지만, 가능할 때 한국어 발음이 가장 자연스럽습니다.
        if self._speak_with_gtts(message):
            return True

        if self._engine is not None:
            self._engine.say(message)
            self._engine.runAndWait()
            return True

        # Linux(spd-say/espeak), macOS(say) 명령이 있으면 fallback으로 사용합니다.
        # 한국어 언어 옵션을 명시해서 영어 음성으로 한국말을 읽는 상황을 줄입니다.
        command_candidates = (
            ("spd-say", ["spd-say", "-l", "ko", message]),
            ("espeak", ["espeak", "-v", "ko", message]),
            ("say", ["say", "-v", "Yuna", message]),
            ("say", ["say", message]),
        )
        for command, args in command_candidates:
            if shutil.which(command):
                subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
        return False

    def _select_korean_voice(self) -> None:
        """pyttsx3에서 한국어 voice가 있으면 우선 선택합니다."""

        if self._engine is None:
            return
        try:
            voices = self._engine.getProperty("voices")
        except Exception:
            return

        for voice in voices:
            voice_text = " ".join(
                str(value).lower()
                for value in (
                    getattr(voice, "id", ""),
                    getattr(voice, "name", ""),
                    getattr(voice, "languages", ""),
                )
            )
            if "ko" in voice_text or "korean" in voice_text:
                self._engine.setProperty("voice", voice.id)
                return

    def _speak_with_gtts(self, message: str) -> bool:
        """gTTS로 한국어 MP3를 만들고 ffplay로 재생합니다."""

        if not shutil.which("ffplay"):
            return False
        try:
            from gtts import gTTS  # type: ignore
        except Exception:
            return False

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
                audio_path = handle.name
            gTTS(text=message, lang="ko").save(audio_path)
            process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            threading.Thread(
                target=self._cleanup_audio_after_playback,
                args=(process, audio_path),
                daemon=True,
            ).start()
            return True
        except Exception:
            return False

    @staticmethod
    def _cleanup_audio_after_playback(process: subprocess.Popen, audio_path: str) -> None:
        """임시 음성 파일을 재생이 끝난 뒤 삭제합니다."""

        process.wait()
        try:
            import os

            os.unlink(audio_path)
        except OSError:
            pass


class FaceOverlapGuidance:
    """얼굴 추적, 겹침 판정, 위치 기반 음성 안내를 한 번에 처리하는 상위 클래스입니다."""

    def __init__(
        self,
        overlap_threshold: float = 0.12,
        resolved_frames: int = 8,
        tracker: Optional[CentroidTracker] = None,
        speech: Optional[SpeechNotifier] = None,
    ) -> None:
        # overlap_threshold 이상 겹치면 "겹쳤다"고 판단합니다.
        self.overlap_threshold = overlap_threshold
        # 겹침이 이 프레임 수만큼 사라져야 같은 겹침 안내를 다시 허용합니다.
        self.resolved_frames = resolved_frames
        self.tracker = tracker or CentroidTracker()
        self.speech = speech or SpeechNotifier(enabled=True)
        self._active_guidance_keys: set[Tuple[int, int]] = set()
        self._clear_frames = 0

    def process(
        self,
        frame: np.ndarray,
        detections: Iterable[DetectionLike | Sequence[int]],
        speak: bool = True,
    ) -> Tuple[Dict[int, TrackedPerson], List[OverlapEvent]]:
        """한 프레임의 검출 결과를 처리하고 추적 결과와 겹침 이벤트를 반환합니다."""

        # 기존 Detection 객체도, 단순 튜플도 모두 BBox로 통일합니다.
        boxes = [_as_bbox(detection) for detection in detections]
        tracks = self.tracker.update(boxes)
        events = self._find_overlap_events(frame.shape, tracks)

        self._update_resolved_state(events)

        # 여러 겹침이 있으면 가장 심한 이벤트 하나만 음성 안내합니다.
        # 같은 두 사람이 계속 겹쳐 있는 동안에는 반복 안내하지 않습니다.
        if speak and events:
            event = events[0]
            guidance_key = self._event_key(event)
            if guidance_key not in self._active_guidance_keys:
                if self.speech.speak(event.message):
                    self._active_guidance_keys.add(guidance_key)

        return tracks, events

    def _update_resolved_state(self, events: List[OverlapEvent]) -> None:
        """겹침이 충분히 사라졌을 때만 다음 안내를 다시 허용합니다."""

        if events:
            self._clear_frames = 0
            return

        self._clear_frames += 1
        if self._clear_frames >= self.resolved_frames:
            self._active_guidance_keys.clear()

    @staticmethod
    def _event_key(event: OverlapEvent) -> Tuple[int, int]:
        """두 사람의 순서와 무관하게 같은 겹침 쌍을 같은 key로 표현합니다."""

        return tuple(sorted((event.object_id, event.other_id)))

    def _find_overlap_events(
        self,
        frame_shape: Tuple[int, ...],
        tracks: Dict[int, TrackedPerson],
    ) -> List[OverlapEvent]:
        """현재 추적 중인 사람들 중 얼굴 박스가 겹친 쌍을 찾습니다."""

        events: List[OverlapEvent] = []
        people = [person for person in tracks.values() if person.disappeared == 0]

        for i, first in enumerate(people):
            for second in people[i + 1 :]:
                ratio = overlap_ratio(first.bbox, second.bbox)
                if ratio < self.overlap_threshold:
                    continue

                # 두 사람 중 화면 중심에서 더 먼 사람에게 움직이라고 안내합니다.
                target = self._choose_guided_person(frame_shape, first, second)
                other = second if target.object_id == first.object_id else first
                position = clock_position(target.centroid, frame_shape)
                message = f"{position}에 있는 분, 옆 사람과 조금 떨어져 주세요."
                events.append(
                    OverlapEvent(
                        object_id=target.object_id,
                        other_id=other.object_id,
                        clock_position=position,
                        message=message,
                        overlap_ratio=ratio,
                    )
                )

        # 음성 안내는 가장 많이 겹친 쌍을 우선으로 합니다.
        events.sort(key=lambda event: event.overlap_ratio, reverse=True)
        return events

    @staticmethod
    def _choose_guided_person(
        frame_shape: Tuple[int, ...],
        first: TrackedPerson,
        second: TrackedPerson,
    ) -> TrackedPerson:
        """겹친 두 사람 중 누구에게 안내할지 선택합니다.

        현재 기준은 단순합니다. 화면 중심에 가까운 사람을 기준으로 두고,
        더 바깥쪽에 있는 사람에게 조금 떨어지라고 안내합니다.
        """

        frame_h, frame_w = frame_shape[:2]
        center = (frame_w / 2.0, frame_h / 2.0)

        def distance_from_center(person: TrackedPerson) -> float:
            return math.hypot(person.centroid[0] - center[0], person.centroid[1] - center[1])

        if distance_from_center(first) >= distance_from_center(second):
            return first
        return second


def make_detector(name: str) -> PersonDetector:
    if name == DETECTOR_HOG:
        return HOGPersonDetector()
    if name == DETECTOR_MEDIAPIPE:
        return MediaPipeFaceDetector()
    if name == DETECTOR_YUNET:
        return YuNetFaceDetector()
    raise RuntimeError(f"Unknown detector: {name}")


def parse_rect(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Rect must be left,top,width,height")
    return tuple(parts)  # type: ignore[return-value]


def clamp_crop(frame: np.ndarray, rect: Optional[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    if rect is None:
        return frame, (0, 0, width, height)

    left, top, crop_w, crop_h = rect
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(left + crop_w, width))
    bottom = max(top + 1, min(top + crop_h, height))
    return frame[top:bottom, left:right], (left, top, right - left, bottom - top)


def compute_people_area_ratio(detections: List[Detection], frame_shape: Tuple[int, ...]) -> float:
    height, width = frame_shape[:2]
    frame_area = max(1, width * height)
    people_area = sum(det.area for det in detections)
    return people_area / frame_area


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


def mode_condition_met(
    mode: str,
    detections: List[Detection],
    frame_shape: Tuple[int, ...],
    person_count: int,
    area_ratio: float,
    target_persons: int,
    ratio_min: float,
    ratio_max: float,
    one_person_ratio_min: float,
    one_person_ratio_max: float,
    center_tolerance: float,
    border_margin_ratio: float,
) -> bool:
    if mode == MODE_PERSON1:
        if person_count != 1:
            return False
        det = detections[0]
        frame_h, frame_w = frame_shape[:2]
        center_alignment = compute_center_alignment(
            detections,
            frame_shape,
            center_tolerance,
        )
        margin_x = frame_w * border_margin_ratio
        margin_y = frame_h * border_margin_ratio
        inside_margin = (
            det.x >= margin_x
            and det.y >= margin_y
            and (det.x + det.w) <= (frame_w - margin_x)
            and (det.y + det.h) <= (frame_h - margin_y)
        )
        within_ratio = one_person_ratio_min <= area_ratio <= one_person_ratio_max
        return center_alignment.aligned and inside_margin and within_ratio
    if mode == MODE_COUNT3:
        return person_count == target_persons
    if mode == MODE_RATIO:
        return ratio_min <= area_ratio <= ratio_max
    return False


def ensure_session_dirs(session_dir: Path) -> Dict[str, Path]:
    paths = {mode: session_dir / mode for mode in MODES}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def order_detections(detections: List[Detection]) -> List[Detection]:
    """Stable left-to-right (then top-to-bottom) order for per-frame labeling.

    This gives each detection a deterministic number (P1, P2, ...) within a
    single frame without any cross-frame tracking, which matches the
    "box + number label" scope for steps 1-3.
    """
    return sorted(detections, key=lambda d: (d.x + d.w / 2.0, d.y + d.h / 2.0))


def draw_overlay(
    frame: np.ndarray,
    crop_rect: Tuple[int, int, int, int],
    detections: List[Detection],
    overlap_tracks: Optional[Dict[int, TrackedPerson]],
    overlap_events: Optional[List[OverlapEvent]],
    mode: str,
    condition_met: bool,
    person_count: int,
    area_ratio: float,
    stable_hits: int,
    stable_frames: int,
    captures_per_mode: Dict[str, int],
    target_persons: int,
    detector_name: str,
    center_alignment: CenterAlignment,
) -> np.ndarray:
    overlay = frame.copy()
    left, top, width, height = crop_rect
    cv2.rectangle(overlay, (left, top), (left + width, top + height), (255, 200, 0), 2)

    count_match = person_count == target_persons
    box_color = (0, 220, 0) if condition_met else (0, 140, 255)

    # Frame labeling: number each detected person P1..Pn (left -> right).
    for index, det in enumerate(order_detections(detections), start=1):
        x1 = left + det.x
        y1 = top + det.y
        x2 = x1 + det.w
        y2 = y1 + det.h
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

        tag = f"P{index} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_top = max(0, y1 - th - 6)
        cv2.rectangle(overlay, (x1, tag_top), (x1 + tw + 6, tag_top + th + 6), box_color, -1)
        cv2.putText(
            overlay,
            tag,
            (x1 + 3, tag_top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if overlap_tracks:
        overlap_events = overlap_events or []
        event_target_ids = {event.object_id for event in overlap_events}
        event_other_ids = {event.other_id for event in overlap_events}
        for object_id, person in overlap_tracks.items():
            x, y, w, h = person.bbox
            x1 = left + x
            y1 = top + y
            x2 = x1 + w
            y2 = y1 + h
            if object_id in event_target_ids:
                track_color = (0, 0, 255)
            elif object_id in event_other_ids:
                track_color = (0, 220, 255)
            else:
                track_color = (255, 220, 0)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), track_color, 1)
            cv2.circle(overlay, (left + person.centroid[0], top + person.centroid[1]), 4, track_color, -1)
            cv2.putText(
                overlay,
                f"ID {object_id}",
                (x1, min(frame.shape[0] - 8, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                track_color,
                2,
                cv2.LINE_AA,
            )

    draw_center_alignment_overlay(cv2, overlay, crop_rect, center_alignment)

    count_state = "MATCH" if count_match else ("OVER" if person_count > target_persons else "UNDER")
    overlap_state = "clear"
    if overlap_events:
        event = overlap_events[0]
        overlap_state = f"ID {event.object_id} move away   ratio={event.overlap_ratio:.2f}"
    info_lines = [
        f"Mode: {mode}   Detector: {detector_name}",
        f"Count: {person_count}/{target_persons} [{count_state}]   Ratio: {area_ratio:.3f}",
        f"Overlap: {overlap_state}",
        f"Stable: {stable_hits}/{stable_frames}   Saved: {captures_per_mode[mode]}",
        "Keys: [1-4] mode  [-/=] target  [c]/[space] capture  [q] quit",
    ]
    if mode == MODE_MANUAL:
        info_lines[2] = f"Manual mode   Saved: {captures_per_mode[mode]}"

    y = 24
    for line in info_lines:
        cv2.putText(
            overlay,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26

    return overlay


def save_capture(
    session_dirs: Dict[str, Path],
    record: CaptureRecord,
    raw_frame: np.ndarray,
    overlay_frame: np.ndarray,
) -> None:
    mode_dir = session_dirs[record.mode]
    raw_path = mode_dir / record.raw_path
    overlay_path = mode_dir / record.overlay_path
    cv2.imwrite(str(raw_path), raw_frame)
    cv2.imwrite(str(overlay_path), overlay_frame)


def append_csv(csv_path: Path, record: CaptureRecord) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                [
                    "capture_id",
                    "mode",
                    "trigger",
                    "timestamp",
                    "elapsed_sec",
                    "person_count",
                    "people_area_ratio",
                    "stable_hits",
                    "raw_path",
                    "overlay_path",
                ]
            )
        writer.writerow(
            [
                record.capture_id,
                record.mode,
                record.trigger,
                f"{record.timestamp:.3f}",
                f"{record.elapsed_sec:.3f}",
                record.person_count,
                f"{record.people_area_ratio:.5f}",
                record.stable_hits,
                record.raw_path,
                record.overlay_path,
            ]
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare manual capture and simple auto-capture heuristics."
    )
    parser.add_argument(
        "--source",
        choices=("webcam", "video", "screen"),
        default="webcam",
        help="Input source type.",
    )
    parser.add_argument(
        "--webcam-index",
        type=int,
        default=0,
        help="Webcam index when --source webcam is used.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="Video file path when --source video is used.",
    )
    parser.add_argument(
        "--screen-region",
        type=parse_rect,
        default=None,
        help="Screen capture region left,top,width,height for mirrored phone window.",
    )
    parser.add_argument(
        "--crop",
        type=parse_rect,
        default=None,
        help="Optional crop inside the input frame: left,top,width,height.",
    )
    parser.add_argument(
        "--detector",
        choices=DETECTORS,
        default=DETECTOR_MEDIAPIPE,
        help="Detection backend: mediapipe (Google face), yunet (OpenCV face), hog (legacy body).",
    )
    parser.add_argument(
        "--start-mode",
        choices=MODES,
        default=MODE_COUNT3,
        help="Initial capture mode.",
    )
    parser.add_argument(
        "--target-persons",
        type=int,
        default=3,
        help="Target head count for count3 mode (step 1: number of people). Adjust live with -/=.",
    )
    parser.add_argument(
        "--ratio-min",
        type=float,
        default=0.18,
        help="Minimum people bbox area ratio for ratio mode.",
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=0.42,
        help="Maximum people bbox area ratio for ratio mode.",
    )
    parser.add_argument(
        "--one-person-ratio-min",
        type=float,
        default=0.10,
        help="Minimum bbox area ratio for person1 mode.",
    )
    parser.add_argument(
        "--one-person-ratio-max",
        type=float,
        default=0.35,
        help="Maximum bbox area ratio for person1 mode.",
    )
    parser.add_argument(
        "--center-tolerance",
        type=float,
        default=0.18,
        help="Allowed center offset ratio for person1 mode.",
    )
    parser.add_argument(
        "--border-margin-ratio",
        type=float,
        default=0.05,
        help="Required free margin ratio on each border for person1 mode.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=10,
        help="Condition must hold for this many consecutive frames before auto-capture.",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=2.0,
        help="Minimum time between captures in auto modes.",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run detection once every N frames. 1 = detect all people in every frame (one-shot).",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.12,
        help="Face overlap threshold for voice guidance. Larger values make guidance less sensitive.",
    )
    parser.add_argument(
        "--disable-overlap-voice",
        action="store_true",
        help="Detect and show face overlap, but do not speak guidance aloud.",
    )
    return parser


def make_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "webcam":
        return VideoSource(args.webcam_index)
    if args.source == "video":
        if not args.video_path:
            raise RuntimeError("--video-path is required when --source video is used.")
        return VideoSource(args.video_path)
    if args.screen_region is None:
        raise RuntimeError(
            "--screen-region left,top,width,height is required when --source screen is used."
        )
    return ScreenSource(args.screen_region)


def main() -> None:
    args = build_arg_parser().parse_args()
    source = make_source(args)
    detector = make_detector(args.detector)
    speech = SpeechNotifier(enabled=not args.disable_overlap_voice)
    overlap_guide = FaceOverlapGuidance(
        overlap_threshold=args.overlap_threshold,
        speech=speech,
    )
    target_persons = max(1, args.target_persons)

    session_name = time.strftime("%Y%m%d-%H%M%S")
    session_dir = OUTPUT_ROOT / session_name
    session_dirs = ensure_session_dirs(session_dir)
    csv_path = session_dir / "captures.csv"

    mode = args.start_mode
    stable_hits = 0
    last_capture_at = 0.0
    captures_per_mode = {name: 0 for name in MODES}
    frame_index = 0
    session_started_at = time.time()
    last_detections: List[Detection] = []

    print("=" * 68)
    print("Session:", session_dir)
    print("Environment: conda activate cv")
    print(f"Detector: {detector.name}   Target persons: {target_persons}")
    print(
        "Overlap guidance:",
        "visual only" if args.disable_overlap_voice else "visual + voice",
        f"(threshold={args.overlap_threshold})",
    )
    print("Modes: 1=manual, 2=person1, 3=count(N), 4=ratio")
    print("Target persons: '-' decrease, '=' increase")
    print("Capture: c or SPACE   Quit: q")
    print("=" * 68)

    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            analysis_frame, crop_rect = clamp_crop(frame, args.crop)

            if frame_index % max(1, args.detect_every) == 0:
                last_detections = detector.detect(analysis_frame)
            detections = last_detections
            overlap_tracks, overlap_events = overlap_guide.process(
                analysis_frame,
                detections,
                speak=not args.disable_overlap_voice,
            )
            person_count = len(detections)
            area_ratio = compute_people_area_ratio(detections, analysis_frame.shape)
            center_alignment = compute_center_alignment(
                detections,
                analysis_frame.shape,
                args.center_tolerance,
            )
            if not center_alignment.aligned:
                speech.speak(center_alignment.overlay_text)

            condition_met = mode_condition_met(
                mode=mode,
                detections=detections,
                frame_shape=analysis_frame.shape,
                person_count=person_count,
                area_ratio=area_ratio,
                target_persons=target_persons,
                ratio_min=args.ratio_min,
                ratio_max=args.ratio_max,
                one_person_ratio_min=args.one_person_ratio_min,
                one_person_ratio_max=args.one_person_ratio_max,
                center_tolerance=args.center_tolerance,
                border_margin_ratio=args.border_margin_ratio,
            )

            if mode == MODE_MANUAL:
                stable_hits = 0
            elif condition_met:
                stable_hits += 1
            else:
                stable_hits = 0

            overlay = draw_overlay(
                frame=frame,
                crop_rect=crop_rect,
                detections=detections,
                overlap_tracks=overlap_tracks,
                overlap_events=overlap_events,
                mode=mode,
                condition_met=condition_met,
                person_count=person_count,
                area_ratio=area_ratio,
                stable_hits=stable_hits,
                stable_frames=args.stable_frames,
                captures_per_mode=captures_per_mode,
                target_persons=target_persons,
                detector_name=detector.name,
                center_alignment=center_alignment,
            )

            now = time.time()
            auto_triggered = (
                mode != MODE_MANUAL
                and stable_hits >= args.stable_frames
                and (now - last_capture_at) >= args.cooldown_sec
            )
            if auto_triggered:
                captures_per_mode[mode] += 1
                capture_id = f"{mode}_{captures_per_mode[mode]:03d}"
                record = CaptureRecord(
                    capture_id=capture_id,
                    mode=mode,
                    trigger="auto",
                    timestamp=now,
                    elapsed_sec=now - session_started_at,
                    person_count=person_count,
                    people_area_ratio=area_ratio,
                    stable_hits=stable_hits,
                    raw_path=f"{capture_id}_raw.png",
                    overlay_path=f"{capture_id}_overlay.png",
                )
                save_capture(session_dirs, record, frame, overlay)
                append_csv(csv_path, record)
                last_capture_at = now
                stable_hits = 0

            cv2.imshow(WINDOW_NAME, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("1"):
                mode = MODE_MANUAL
                stable_hits = 0
            elif key == ord("2"):
                mode = MODE_PERSON1
                stable_hits = 0
            elif key == ord("3"):
                mode = MODE_COUNT3
                stable_hits = 0
            elif key == ord("4"):
                mode = MODE_RATIO
                stable_hits = 0
            elif key == ord("-"):
                target_persons = max(1, target_persons - 1)
                stable_hits = 0
                print(f"[target] persons = {target_persons}")
            elif key in (ord("="), ord("+")):
                target_persons += 1
                stable_hits = 0
                print(f"[target] persons = {target_persons}")
            elif key in (ord("c"), 32):
                captures_per_mode[mode] += 1
                now = time.time()
                capture_id = f"{mode}_{captures_per_mode[mode]:03d}"
                record = CaptureRecord(
                    capture_id=capture_id,
                    mode=mode,
                    trigger="manual-key",
                    timestamp=now,
                    elapsed_sec=now - session_started_at,
                    person_count=person_count,
                    people_area_ratio=area_ratio,
                    stable_hits=stable_hits,
                    raw_path=f"{capture_id}_raw.png",
                    overlay_path=f"{capture_id}_overlay.png",
                )
                save_capture(session_dirs, record, frame, overlay)
                append_csv(csv_path, record)
                last_capture_at = now

            frame_index += 1
    finally:
        source.close()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
