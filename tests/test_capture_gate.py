"""count3 품질 게이트(mode_condition_met) 단위 테스트.

shot_capture_experiment 가 cv2/numpy 를 import 하므로 conda cv 환경에서 실행한다:
    conda activate cv
    python -m pytest tests/test_capture_gate.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shot_capture_experiment import (  # noqa: E402
    MODE_COUNT3,
    Detection,
    mode_condition_met,
)

FRAME_SHAPE = (480, 640, 3)


def _three_people():
    return [Detection(x=300, y=220, w=40, h=40, score=0.9) for _ in range(3)]


def _gate(person_count, *, inside, no_overlap, centered, require_quality=True):
    return mode_condition_met(
        mode=MODE_COUNT3,
        detections=_three_people(),
        frame_shape=FRAME_SHAPE,
        person_count=person_count,
        area_ratio=0.2,
        target_persons=3,
        ratio_min=0.1,
        ratio_max=0.5,
        one_person_ratio_min=0.1,
        one_person_ratio_max=0.5,
        center_tolerance=0.2,
        border_margin_ratio=0.05,
        inside_frame=inside,
        no_overlap=no_overlap,
        centered=centered,
        require_quality=require_quality,
    )


def test_count3_all_conditions_met_triggers():
    assert _gate(3, inside=True, no_overlap=True, centered=True) is True


def test_count3_blocks_when_face_clipped():
    assert _gate(3, inside=False, no_overlap=True, centered=True) is False


def test_count3_blocks_when_overlapping():
    assert _gate(3, inside=True, no_overlap=False, centered=True) is False


def test_count3_blocks_when_off_center():
    assert _gate(3, inside=True, no_overlap=True, centered=False) is False


def test_count3_blocks_on_wrong_count_even_if_quality_ok():
    assert _gate(2, inside=True, no_overlap=True, centered=True) is False


def test_count3_legacy_mode_ignores_quality():
    # --disable-count3-quality => require_quality=False => 머릿수만 본다.
    assert _gate(3, inside=False, no_overlap=False, centered=False, require_quality=False) is True
    assert _gate(2, inside=True, no_overlap=True, centered=True, require_quality=False) is False


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
