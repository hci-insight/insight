from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from speech import SpeechNotifier
from tracking import CentroidTracker, TrackedPerson
from utils import BBox, DetectionLike, _as_bbox


@dataclass
class OutOfFrameEvent:
    """얼굴이 화면 경계 밖으로 잘렸을 때 안내에 필요한 정보를 담습니다."""

    object_id: int
    directions: List[str]
    message: str
    clip_ratio: float


class FaceOutOfFrameGuidance:
    """얼굴 박스가 프레임 경계에 잘렸을 때 위치 이동을 안내하는 클래스입니다.

    clip_threshold 이상의 얼굴 면적이 잘렸을 때 음성 안내를 제공합니다.
    """

    def __init__(
        self,
        clip_threshold: float = 0.10,
        resolved_frames: int = 8,
        tracker: Optional[CentroidTracker] = None,
        speech: Optional[SpeechNotifier] = None,
        id_cooldown_sec: float = 5.0,
    ) -> None:
        self.clip_threshold = clip_threshold
        self.resolved_frames = resolved_frames
        self.tracker = tracker or CentroidTracker()
        self.speech = speech or SpeechNotifier(enabled=True)
        self._last_spoken: Dict[int, float] = {}
        self._id_cooldown_sec = id_cooldown_sec

    def process(
        self,
        frame: np.ndarray,
        detections: Iterable[Union[DetectionLike, Sequence[int]]],
        speak: bool = True,
    ) -> Tuple[Dict[int, TrackedPerson], List[OutOfFrameEvent]]:
        """한 프레임의 검출 결과를 처리하고 추적 결과와 이탈 이벤트를 반환합니다."""
        boxes = [_as_bbox(d) for d in detections]
        tracks = self.tracker.update(boxes)
        events = self._find_outofframe_events(frame.shape, tracks)

        if speak and events:
            event = events[0]
            now = time.time()
            if now - self._last_spoken.get(event.object_id, 0.0) >= self._id_cooldown_sec:
                if self.speech.speak(event.message):
                    self._last_spoken[event.object_id] = now

        return tracks, events

    def _find_outofframe_events(
        self,
        frame_shape: Tuple[int, ...],
        tracks: Dict[int, TrackedPerson],
    ) -> List[OutOfFrameEvent]:
        events: List[OutOfFrameEvent] = []
        frame_h, frame_w = frame_shape[:2]
        people = [p for p in tracks.values() if p.disappeared == 0]

        for person in people:
            x, y, w, h = person.bbox
            face_area = max(1, w * h)

            # 대부분의 검출기는 bbox를 프레임 내로 클리핑해서 반환하므로,
            # 얼굴 크기의 15%를 edge margin으로 사용해 "가장자리 근접 = 이탈"로 간주합니다.
            edge_x = max(10, int(w * 0.15))
            edge_y = max(10, int(h * 0.15))

            clip_left   = max(0, edge_x - x)
            clip_top    = max(0, edge_y - y)
            clip_right  = max(0, (x + w) - (frame_w - edge_x))
            clip_bottom = max(0, (y + h) - (frame_h - edge_y))

            visible_area = max(0, w - clip_left - clip_right) * max(0, h - clip_top - clip_bottom)
            clip_ratio = 1.0 - visible_area / face_area

            if clip_ratio < self.clip_threshold:
                continue

            # 잘린 방향을 수집해 "이쪽으로 이동하라"는 메시지로 변환합니다.
            directions: List[str] = []
            if clip_left > 0:
                directions.append("오른쪽")
            if clip_right > 0:
                directions.append("왼쪽")
            if clip_top > 0:
                directions.append("아래")
            if clip_bottom > 0:
                directions.append("위")

            if not directions:
                continue

            direction_str = " 또는 ".join(directions)
            message = f"화면 가장자리에 있는 분, {direction_str}으로 조금 이동해 주세요."
            events.append(
                OutOfFrameEvent(
                    object_id=person.object_id,
                    directions=directions,
                    message=message,
                    clip_ratio=clip_ratio,
                )
            )

        events.sort(key=lambda e: e.clip_ratio, reverse=True)
        return events
