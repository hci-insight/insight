from __future__ import annotations

from typing import Optional, Set

from speech import SpeechNotifier

# 촬영 직전 카운트다운에 사용할 한국어 숫자입니다.
KOREAN_NUMBERS = {1: "하나", 2: "둘", 3: "셋", 4: "넷", 5: "다섯"}

# update()가 돌려주는 상태 문자열입니다.
STATUS_IDLE = "idle"          # 카운트다운 대기 (촬영 조건 미충족)
STATUS_COUNTING = "counting"  # 카운트다운 진행 중
STATUS_ABORTED = "aborted"    # 진행 중 조건이 깨져 취소됨
STATUS_CAPTURE = "capture"    # 카운트다운 완료 → 지금 촬영


class CountdownController:
    """촬영 조건이 안정적으로 충족되면 "하나·둘·셋" 음성 카운트다운을 진행하고,
    완료 시 촬영 신호(STATUS_CAPTURE)를 돌려주는 상태 머신입니다.

    프레임 루프에서 매 프레임 ``update(ready, now)`` 를 호출합니다.
    - ``ready`` : 촬영 조건 + 안정 프레임 + 쿨다운을 모두 만족하는지 여부
    - ``now``   : 현재 시각(time.time()). 주입식이라 단위 테스트가 쉽습니다.

    진행 중에 ``ready`` 가 False가 되면 즉시 취소(STATUS_ABORTED)하고 리셋합니다.
    """

    def __init__(
        self,
        speech: Optional[SpeechNotifier] = None,
        steps: int = 3,
        step_sec: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.speech = speech or SpeechNotifier(enabled=True)
        self.steps = max(1, int(steps))
        self.step_sec = max(0.1, float(step_sec))
        self.enabled = enabled
        self.active = False
        self.start_at = 0.0
        self._spoken_steps: Set[int] = set()

    @property
    def total_sec(self) -> float:
        return self.steps * self.step_sec

    def remaining_steps(self, now: float) -> int:
        """오버레이 표시용. 카운트다운 중 남은 숫자(예: 3,2,1)를 반환합니다."""
        if not self.active:
            return 0
        elapsed = now - self.start_at
        done = int(elapsed // self.step_sec)
        return max(0, self.steps - done)

    def reset(self) -> None:
        self.active = False
        self.start_at = 0.0
        self._spoken_steps = set()

    def update(self, ready: bool, now: float) -> str:
        # 카운트다운을 끈 경우: 조건 충족 즉시 촬영, 음성 없음.
        if not self.enabled:
            self.active = False
            return STATUS_CAPTURE if ready else STATUS_IDLE

        if not self.active:
            if not ready:
                return STATUS_IDLE
            # 카운트다운 시작.
            self.active = True
            self.start_at = now
            self._spoken_steps = set()

        # 진행 중 조건이 깨지면 취소.
        if not ready:
            self.reset()
            return STATUS_ABORTED

        elapsed = now - self.start_at

        # 경과 시간에 도달한 숫자를 한 번씩만 음성으로 읽습니다.
        # 숫자 i 는 t = (i-1) * step_sec 에 읽습니다 (i = 1..steps).
        current_step = min(self.steps, int(elapsed // self.step_sec) + 1)
        for i in range(1, current_step + 1):
            if i not in self._spoken_steps:
                self.speech.speak(KOREAN_NUMBERS.get(i, str(i)), force=True)
                self._spoken_steps.add(i)

        if elapsed >= self.total_sec:
            self.reset()
            return STATUS_CAPTURE

        return STATUS_COUNTING
