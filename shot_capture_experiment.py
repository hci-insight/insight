from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

        self._mp = mp
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
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
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
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
        center_x = det.x + det.w / 2.0
        center_y = det.y + det.h / 2.0
        target_x = frame_w / 2.0
        target_y = frame_h / 2.0
        within_center_x = abs(center_x - target_x) <= frame_w * center_tolerance
        within_center_y = abs(center_y - target_y) <= frame_h * center_tolerance
        margin_x = frame_w * border_margin_ratio
        margin_y = frame_h * border_margin_ratio
        inside_margin = (
            det.x >= margin_x
            and det.y >= margin_y
            and (det.x + det.w) <= (frame_w - margin_x)
            and (det.y + det.h) <= (frame_h - margin_y)
        )
        within_ratio = one_person_ratio_min <= area_ratio <= one_person_ratio_max
        return within_center_x and within_center_y and inside_margin and within_ratio
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
    mode: str,
    condition_met: bool,
    person_count: int,
    area_ratio: float,
    stable_hits: int,
    stable_frames: int,
    captures_per_mode: Dict[str, int],
    target_persons: int,
    detector_name: str,
) -> np.ndarray:
    overlay = frame.copy()
    left, top, width, height = crop_rect
    cv2.rectangle(overlay, (left, top), (left + width, top + height), (255, 200, 0), 2)

    count_match = person_count == target_persons
    box_color = (0, 220, 0) if condition_met else (0, 140, 255)
    label_color = (0, 220, 0) if count_match else (0, 140, 255)

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

    count_state = "MATCH" if count_match else ("OVER" if person_count > target_persons else "UNDER")
    info_lines = [
        f"Mode: {mode}   Detector: {detector_name}",
        f"Count: {person_count}/{target_persons} [{count_state}]   Ratio: {area_ratio:.3f}",
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
            person_count = len(detections)
            area_ratio = compute_people_area_ratio(detections, analysis_frame.shape)
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
                mode=mode,
                condition_met=condition_met,
                person_count=person_count,
                area_ratio=area_ratio,
                stable_hits=stable_hits,
                stable_frames=args.stable_frames,
                captures_per_mode=captures_per_mode,
                target_persons=target_persons,
                detector_name=detector.name,
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
