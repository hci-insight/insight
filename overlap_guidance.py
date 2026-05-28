from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from speech import SpeechNotifier
from tracking import CentroidTracker, TrackedPerson
from utils import BBox, DetectionLike, Point, _as_bbox, clock_position, overlap_ratio


@dataclass
class OverlapEvent:
    """얼굴 겹침이 감지되었을 때 안내에 필요한 정보를 담습니다."""

    object_id: int
    other_id: int
    clock_position: str
    message: str
    overlap_ratio: float


class FaceOverlapGuidance:
    """얼굴 추적, 겹침 판정, 위치 기반 음성 안내를 한 번에 처리하는 상위 클래스입니다."""

    def __init__(
        self,
        overlap_threshold: float = 0.12,
        resolved_frames: int = 8,
        tracker: Optional[CentroidTracker] = None,
        speech: Optional[SpeechNotifier] = None,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.resolved_frames = resolved_frames
        self.tracker = tracker or CentroidTracker()
        self.speech = speech or SpeechNotifier(enabled=True)
        self._active_guidance_keys: set[Tuple[int, int]] = set()
        self._clear_frames = 0

    def process(
        self,
        frame: np.ndarray,
        detections: Iterable[Union[DetectionLike, Sequence[int]]],
        speak: bool = True,
    ) -> Tuple[Dict[int, TrackedPerson], List[OverlapEvent]]:
        """한 프레임의 검출 결과를 처리하고 추적 결과와 겹침 이벤트를 반환합니다."""
        boxes = [_as_bbox(d) for d in detections]
        tracks = self.tracker.update(boxes)
        events = self._find_overlap_events(frame.shape, tracks)

        self._update_resolved_state(events)

        if speak and events:
            event = events[0]
            guidance_key = self._event_key(event)
            if guidance_key not in self._active_guidance_keys:
                if self.speech.speak(event.message):
                    self._active_guidance_keys.add(guidance_key)

        return tracks, events

    def _update_resolved_state(self, events: List[OverlapEvent]) -> None:
        if events:
            self._clear_frames = 0
            return
        self._clear_frames += 1
        if self._clear_frames >= self.resolved_frames:
            self._active_guidance_keys.clear()

    @staticmethod
    def _event_key(event: OverlapEvent) -> Tuple[int, int]:
        return tuple(sorted((event.object_id, event.other_id)))  # type: ignore[return-value]

    def _find_overlap_events(
        self,
        frame_shape: Tuple[int, ...],
        tracks: Dict[int, TrackedPerson],
    ) -> List[OverlapEvent]:
        events: List[OverlapEvent] = []
        people = [p for p in tracks.values() if p.disappeared == 0]

        for i, first in enumerate(people):
            for second in people[i + 1:]:
                ratio = overlap_ratio(first.bbox, second.bbox)
                if ratio < self.overlap_threshold:
                    continue

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

        events.sort(key=lambda e: e.overlap_ratio, reverse=True)
        return events

    @staticmethod
    def _choose_guided_person(
        frame_shape: Tuple[int, ...],
        first: TrackedPerson,
        second: TrackedPerson,
    ) -> TrackedPerson:
        """겹친 두 사람 중 화면 중심에서 더 먼 사람을 안내 대상으로 선택합니다."""
        frame_h, frame_w = frame_shape[:2]
        cx, cy = frame_w / 2.0, frame_h / 2.0

        def dist(person: TrackedPerson) -> float:
            return math.hypot(person.centroid[0] - cx, person.centroid[1] - cy)

        return first if dist(first) >= dist(second) else second
