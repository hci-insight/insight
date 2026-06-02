from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from camera_guidance import (
    CenterAlignment,
    compute_center_alignment,
    draw_center_alignment_overlay,
)
from detection import (
    DETECTOR_MEDIAPIPE,
    DETECTORS,
    Detection,
    PersonDetector,
    make_detector,
)
from overlap_guidance import FaceOverlapGuidance, OverlapEvent
from speech import SpeechNotifier
from tracking import TrackedPerson

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
WINDOW_NAME = "HCI Shot Capture Experiment"

MODE_MANUAL = "manual"
MODE_PERSON1 = "person1"
MODE_COUNT3 = "count3"
MODE_RATIO = "ratio"
MODES = (MODE_MANUAL, MODE_PERSON1, MODE_COUNT3, MODE_RATIO)


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


def parse_rect(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Rect must be left,top,width,height")
    return tuple(parts)  # type: ignore[return-value]


def clamp_crop(
    frame: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    if rect is None:
        return frame, (0, 0, width, height)

    left, top, crop_w, crop_h = rect
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(left + crop_w, width))
    bottom = max(top + 1, min(top + crop_h, height))
    return frame[top:bottom, left:right], (left, top, right - left, bottom - top)


def compute_people_area_ratio(
    detections: List[Detection],
    frame_shape: Tuple[int, ...],
) -> float:
    height, width = frame_shape[:2]
    frame_area = max(1, width * height)
    return sum(det.area for det in detections) / frame_area


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
    inside_frame: bool = True,
    no_overlap: bool = True,
    centered: bool = True,
    require_quality: bool = True,
) -> bool:
    if mode == MODE_PERSON1:
        if person_count != 1:
            return False
        det = detections[0]
        frame_h, frame_w = frame_shape[:2]
        center_alignment = compute_center_alignment(detections, frame_shape, center_tolerance)
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
        if person_count != target_persons:
            return False
        # 인원수가 맞아도 화면내부/중심/겹침없음 세 품질조건을 모두 만족해야 촬영한다.
        # (--disable-count3-quality 로 require_quality=False 가 되면 옛 머릿수-only 동작)
        if require_quality:
            return inside_frame and no_overlap and centered
        return True
    if mode == MODE_RATIO:
        return ratio_min <= area_ratio <= ratio_max
    return False


def ensure_session_dirs(session_dir: Path) -> Dict[str, Path]:
    paths = {mode: session_dir / mode for mode in MODES}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def order_detections(detections: List[Detection]) -> List[Detection]:
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
            cv2.circle(
                overlay,
                (left + person.centroid[0], top + person.centroid[1]),
                4,
                track_color,
                -1,
            )
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
        center_alignment.overlay_text,
        f"Overlap: {overlap_state}",
        f"Stable: {stable_hits}/{stable_frames}   Saved: {captures_per_mode[mode]}",
        "Keys: [1-4] mode  [-/=] target  [c]/[space] capture  [q] quit",
    ]
    if mode == MODE_MANUAL:
        info_lines[3] = f"Manual mode   Saved: {captures_per_mode[mode]}"

    y = 24
    for line in info_lines:
        cv2.putText(
            overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA
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
    cv2.imwrite(str(mode_dir / record.raw_path), raw_frame)
    cv2.imwrite(str(mode_dir / record.overlay_path), overlay_frame)


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
        help="Target head count for count3 mode. Adjust live with -/=.",
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
        help="Run detection once every N frames.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.12,
        help="Face overlap threshold for voice guidance.",
    )
    parser.add_argument(
        "--disable-overlap-voice",
        action="store_true",
        dest="disable_guidance_voice",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-guidance-voice",
        action="store_true",
        help="Show guidance visually, but do not speak feedback aloud.",
    )
    parser.add_argument(
        "--disable-count3-quality",
        action="store_true",
        help="Revert count3 to head-count-only (skip inside-frame/center/overlap gate).",
    )
    parser.add_argument(
        "--no-countdown",
        action="store_true",
        help="Capture immediately when conditions are met (skip the spoken countdown).",
    )
    parser.add_argument(
        "--countdown-steps",
        type=int,
        default=3,
        help="Number of spoken countdown steps before capture (하나/둘/셋 = 3).",
    )
    parser.add_argument(
        "--countdown-step-sec",
        type=float,
        default=1.0,
        help="Seconds between countdown steps.",
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

