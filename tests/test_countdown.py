"""CountdownController 상태 머신 단위 테스트.

cv2/numpy 없이 순수 로직만 검증한다. 실행:
    python -m pytest tests/test_countdown.py
또는 의존성 없이:
    python tests/test_countdown.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from countdown import (  # noqa: E402
    STATUS_ABORTED,
    STATUS_CAPTURE,
    STATUS_COUNTING,
    STATUS_IDLE,
    CountdownController,
)


class FakeSpeech:
    """speak() 호출만 기록하는 가짜 음성 객체."""

    def __init__(self) -> None:
        self.spoken = []

    def speak(self, message: str, force: bool = False) -> bool:
        self.spoken.append(message)
        return True


def test_idle_when_not_ready():
    c = CountdownController(speech=FakeSpeech(), steps=3, step_sec=1.0)
    assert c.update(ready=False, now=0.0) == STATUS_IDLE
    assert not c.active


def test_full_countdown_speaks_and_captures():
    speech = FakeSpeech()
    c = CountdownController(speech=speech, steps=3, step_sec=1.0)

    assert c.update(ready=True, now=0.0) == STATUS_COUNTING   # 하나
    assert c.update(ready=True, now=1.0) == STATUS_COUNTING   # 둘
    assert c.update(ready=True, now=2.0) == STATUS_COUNTING   # 셋
    assert c.update(ready=True, now=3.0) == STATUS_CAPTURE    # 촬영

    assert speech.spoken == ["하나", "둘", "셋"]
    assert not c.active  # 촬영 후 리셋


def test_each_number_spoken_once():
    speech = FakeSpeech()
    c = CountdownController(speech=speech, steps=3, step_sec=1.0)
    # 같은 구간에서 여러 프레임이 돌아도 숫자는 한 번씩만.
    c.update(ready=True, now=0.0)
    c.update(ready=True, now=0.3)
    c.update(ready=True, now=0.6)
    c.update(ready=True, now=1.0)
    assert speech.spoken == ["하나", "둘"]


def test_abort_when_condition_breaks():
    speech = FakeSpeech()
    c = CountdownController(speech=speech, steps=3, step_sec=1.0)
    c.update(ready=True, now=0.0)   # 하나
    c.update(ready=True, now=1.0)   # 둘
    assert c.update(ready=False, now=1.5) == STATUS_ABORTED
    assert not c.active
    assert speech.spoken == ["하나", "둘"]


def test_restart_after_abort():
    speech = FakeSpeech()
    c = CountdownController(speech=speech, steps=3, step_sec=1.0)
    c.update(ready=True, now=0.0)
    c.update(ready=False, now=0.5)        # abort
    assert c.update(ready=True, now=2.0) == STATUS_COUNTING  # 새 카운트다운 시작
    assert speech.spoken == ["하나", "하나"]


def test_disabled_captures_immediately_without_speaking():
    speech = FakeSpeech()
    c = CountdownController(speech=speech, steps=3, step_sec=1.0, enabled=False)
    assert c.update(ready=False, now=0.0) == STATUS_IDLE
    assert c.update(ready=True, now=0.0) == STATUS_CAPTURE
    assert speech.spoken == []


def test_remaining_steps():
    c = CountdownController(speech=FakeSpeech(), steps=3, step_sec=1.0)
    assert c.remaining_steps(0.0) == 0   # 아직 비활성
    c.update(ready=True, now=0.0)
    assert c.remaining_steps(0.0) == 3
    c.update(ready=True, now=1.0)
    assert c.remaining_steps(1.0) == 2
    c.update(ready=True, now=2.0)
    assert c.remaining_steps(2.0) == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
