from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence, Tuple


class DetectionLike(Protocol):
    x: int
    y: int
    w: int
    h: int


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
    else:
        overlay_directions: List[str] = []
        if not aligned_x:
            if offset_x < 0:
                overlay_directions.append("left")
            else:
                overlay_directions.append("right")
        if not aligned_y:
            if offset_y < 0:
                overlay_directions.append("up")
            else:
                overlay_directions.append("down")
        overlay_text = f"Turn camera {' and '.join(overlay_directions)}"

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
    )
