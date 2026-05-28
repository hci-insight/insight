from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from utils import BBox, Point, bbox_centroid


@dataclass
class TrackedPerson:
    """Centroid Tracking으로 유지되는 한 사람의 현재 상태입니다."""

    object_id: int
    bbox: BBox
    centroid: Point
    disappeared: int = 0


class CentroidTracker:
    """프레임이 바뀌어도 같은 사람에게 같은 ID를 유지하는 간단한 추적기입니다."""

    def __init__(self, max_disappeared: int = 20, max_distance: float = 80.0) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 1
        self.objects: Dict[int, TrackedPerson] = {}

    def update(self, boxes: Iterable[BBox]) -> Dict[int, TrackedPerson]:
        """새 프레임의 얼굴 박스 목록을 받아 추적 ID를 갱신합니다."""
        input_boxes = list(boxes)
        input_centroids = [bbox_centroid(box) for box in input_boxes]

        if not input_boxes:
            for object_id in list(self.objects):
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    del self.objects[object_id]
            return dict(self.objects)

        if not self.objects:
            for box, centroid in zip(input_boxes, input_centroids):
                self._register(box, centroid)
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]
        distances = self._distance_matrix(object_centroids, input_centroids)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = TrackedPerson(
                object_id=object_id,
                bbox=input_boxes[col],
                centroid=input_centroids[col],
                disappeared=0,
            )
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(object_ids))) - used_rows:
            object_id = object_ids[row]
            self.objects[object_id].disappeared += 1
            if self.objects[object_id].disappeared > self.max_disappeared:
                del self.objects[object_id]

        for col in set(range(len(input_boxes))) - used_cols:
            self._register(input_boxes[col], input_centroids[col])

        return dict(self.objects)

    def _register(self, bbox: BBox, centroid: Point) -> None:
        object_id = self.next_object_id
        self.objects[object_id] = TrackedPerson(object_id, bbox, centroid)
        self.next_object_id += 1

    @staticmethod
    def _distance_matrix(a: List[Point], b: List[Point]) -> np.ndarray:
        matrix = np.zeros((len(a), len(b)), dtype="float32")
        for row, (ax, ay) in enumerate(a):
            for col, (bx, by) in enumerate(b):
                matrix[row, col] = math.hypot(ax - bx, ay - by)
        return matrix
