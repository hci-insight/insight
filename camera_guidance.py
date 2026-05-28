from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from speech import SpeechNotifier
from utils import DetectionLike


@dataclass
class CenterAlignment:
    has_target: bool
    aligned: bool
    target_center_x: float
    target_center_y: float
    frame_center_x: float
    frame_center_y: float
    offset_x: float
    offset_y: float
    overlay_text: str
    voice_message: str
    voice_key: str


def draw_center_alignment_overlay(
    cv2_module: Any,
    overlay: Any,
    crop_rect: Tuple[int, int, int, int],
    center_alignment: CenterAlignment,
) -> None:
    left, top, _, _ = crop_rect
    guide_color = (0, 220, 0) if center_alignment.aligned else (0, 140, 255)
    frame_center = (
        left + int(round(center_alignment.frame_center_x)),
        top + int(round(center_alignment.frame_center_y)),
    )
    cv2_module.drawMarker(
        overlay,
        frame_center,
        guide_color,
        markerType=cv2_module.MARKER_CROSS,
        markerSize=34,
        thickness=2,
        line_type=cv2_module.LINE_AA,
    )

    if center_alignment.has_target:
        target_center = (
            left + int(round(center_alignment.target_center_x)),
            top + int(round(center_alignment.target_center_y)),
        )
        cv2_module.circle(overlay, target_center, 6, (255, 120, 0), -1)
        cv2_module.line(
            overlay,
            frame_center,
            target_center,
            guide_color,
            2,
            cv2_module.LINE_AA,
        )

    text = center_alignment.overlay_text
    if text:
        font = cv2_module.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2_module.getTextSize(text, font, scale, thickness)
        text_x = left + 12
        text_y = top + int(round(center_alignment.frame_center_y)) + 50
        box_tl = (text_x - 8, text_y - text_h - 8)
        box_br = (text_x + text_w + 8, text_y + 8)
        cv2_module.rectangle(overlay, box_tl, box_br, (0, 0, 0), -1)
        cv2_module.putText(
            overlay,
            text,
            (text_x, text_y),
            font,
            scale,
            guide_color,
            thickness,
            cv2_module.LINE_AA,
        )


def compute_center_alignment(
    detections: Sequence[DetectionLike],
    frame_shape: Sequence[int],
    center_tolerance: float,
) -> CenterAlignment:
    frame_h, frame_w = frame_shape[:2]
    frame_center_x = frame_w / 2.0
    frame_center_y = frame_h / 2.0
    tolerance_x = frame_w * center_tolerance
    tolerance_y = frame_h * center_tolerance

    if not detections:
        return CenterAlignment(
            has_target=False,
            aligned=False,
            target_center_x=frame_center_x,
            target_center_y=frame_center_y,
            frame_center_x=frame_center_x,
            frame_center_y=frame_center_y,
            offset_x=0.0,
            offset_y=0.0,
            overlay_text="No target",
            voice_message="",
            voice_key="",
        )

    left = min(det.x for det in detections)
    top = min(det.y for det in detections)
    right = max(det.x + det.w for det in detections)
    bottom = max(det.y + det.h for det in detections)
    target_center_x = (left + right) / 2.0
    target_center_y = (top + bottom) / 2.0
    offset_x = target_center_x - frame_center_x
    offset_y = target_center_y - frame_center_y
    aligned_x = abs(offset_x) <= tolerance_x
    aligned_y = abs(offset_y) <= tolerance_y
    aligned = aligned_x and aligned_y

    if aligned:
        overlay_text = "Centered"
        voice_message = ""
        voice_key = ""
    else:
        overlay_directions: List[str] = []
        voice_directions: List[str] = []
        if not aligned_x:
            overlay_directions.append("left" if offset_x < 0 else "right")
            voice_directions.append("왼쪽" if offset_x < 0 else "오른쪽")
        if not aligned_y:
            overlay_directions.append("up" if offset_y < 0 else "down")
            voice_directions.append("위쪽" if offset_y < 0 else "아래쪽")
        overlay_text = f"Turn camera {' and '.join(overlay_directions)}"
        voice_direction_text = "과 ".join(voice_directions)
        voice_message = (
            f"인물들의 중심이 {voice_direction_text}으로 치우쳐 있어요. "
            f"카메라를 {voice_direction_text}으로 조금 돌려 주세요."
        )
        voice_key = ",".join(overlay_directions)

    return CenterAlignment(
        has_target=True,
        aligned=aligned,
        target_center_x=target_center_x,
        target_center_y=target_center_y,
        frame_center_x=frame_center_x,
        frame_center_y=frame_center_y,
        offset_x=offset_x,
        offset_y=offset_y,
        overlay_text=overlay_text,
        voice_message=voice_message,
        voice_key=voice_key,
    )


class CenterAlignmentVoiceGuidance:
    """인물들의 무게중심이 프레임 중심에서 벗어났을 때 음성으로 안내합니다."""

    def __init__(
        self,
        speech: SpeechNotifier | None = None,
        resolved_frames: int = 8,
    ) -> None:
        # 같은 중심 이탈 상태가 이어지는 동안 안내가 계속 반복되지 않도록 상태를 기억합니다.
        self.speech = speech or SpeechNotifier(enabled=True)
        self.resolved_frames = resolved_frames
        self._active_key = ""
        self._resolved_count = 0

    def process(self, center_alignment: CenterAlignment, speak: bool = True) -> bool:
        """중심 정렬 결과를 보고 필요한 경우 한 번만 음성 안내를 출력합니다."""

        if not speak:
            return False

        if not center_alignment.has_target or center_alignment.aligned:
            self._resolved_count += 1
            if self._resolved_count >= self.resolved_frames:
                self._active_key = ""
            return False

        self._resolved_count = 0
        if not center_alignment.voice_message or center_alignment.voice_key == self._active_key:
            return False

        spoken = self.speech.speak(center_alignment.voice_message)
        if spoken:
            self._active_key = center_alignment.voice_key
        return spoken
