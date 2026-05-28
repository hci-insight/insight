from __future__ import annotations

from dataclasses import dataclass, field
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
    ) -> None:
        # clip_threshold: 얼굴 면적의 몇 % 이상이 잘려야 안내를 트리거할지 결정합니다.
        self.clip_threshold = clip_threshold
        self.resolved_frames = resolved_frames
        self.tracker = tracker or CentroidTracker()
        self.speech = speech or SpeechNotifier(enabled=True)
        self._active_guidance_ids: set[int] = set()
        self._clear_frames = 0

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

        self._update_resolved_state(events)

        if speak and events:
            event = events[0]
            if event.object_id not in self._active_guidance_ids:
                if self.speech.speak(event.message):
                    self._active_guidance_ids.add(event.object_id)

        return tracks, events

    def _update_resolved_state(self, events: List[OutOfFrameEvent]) -> None:
        if events:
            self._clear_frames = 0
            return
        self._clear_frames += 1
        if self._clear_frames >= self.resolved_frames:
            self._active_guidance_ids.clear()

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

            # 각 방향으로 잘린 픽셀 수를 계산합니다.
            clip_left = max(0, -x)
            clip_top = max(0, -y)
            clip_right = max(0, (x + w) - frame_w)
            clip_bottom = max(0, (y + h) - frame_h)

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
