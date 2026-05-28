from __future__ import annotations

import math
from typing import Protocol, Sequence, Tuple, Union

BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


class DetectionLike(Protocol):
    x: int
    y: int
    w: int
    h: int


def _as_bbox(detection: Union[DetectionLike, Sequence[int]]) -> BBox:
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
    x, y, w, h = bbox
    return int(x + w / 2), int(y + h / 2)


def bbox_area(bbox: BBox) -> int:
    return max(0, bbox[2]) * max(0, bbox[3])


def overlap_ratio(a: BBox, b: BBox) -> float:
    """교집합 / 더 작은 박스 면적으로 겹침 정도를 계산합니다."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    inter_left = max(ax, bx)
    inter_top = max(ay, by)
    inter_right = min(ax + aw, bx + bw)
    inter_bottom = min(ay + ah, by + bh)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    smaller_area = max(1, min(bbox_area(a), bbox_area(b)))
    return inter_area / smaller_area


def clock_position(
    centroid: Point,
    frame_shape: Tuple[int, ...],
    center_radius_ratio: float = 0.20,
) -> str:
    """프레임 중심 기준으로 시계 방향 위치 문자열을 반환합니다."""
    frame_h, frame_w = frame_shape[:2]
    cx, cy = centroid
    dx = cx - frame_w / 2.0
    dy = cy - frame_h / 2.0

    center_radius = min(frame_w, frame_h) * center_radius_ratio
    if math.hypot(dx, dy) <= center_radius:
        return "가운데"

    angle = math.degrees(math.atan2(dx, -dy))
    if angle < 0:
        angle += 360.0
    hour = int(round(angle / 30.0)) % 12
    hour = 12 if hour == 0 else hour
    return f"{hour}시"
