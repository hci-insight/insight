from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time


class SpeechNotifier:
    """로컬 TTS가 가능하면 한국어 안내 문장을 음성으로 읽어주는 클래스입니다."""

    def __init__(self, enabled: bool = True, cooldown_sec: float = 2.0) -> None:
        self.enabled = enabled
        self.cooldown_sec = cooldown_sec
        self.last_spoken_at = 0.0
        self.last_message = ""
        self._pyttsx3 = None
        self._engine = None

        if not enabled:
            return
        try:
            import pyttsx3  # type: ignore

            self._pyttsx3 = pyttsx3
            self._engine = pyttsx3.init()
            self._select_korean_voice()
        except Exception:
            self._pyttsx3 = None
            self._engine = None

    def speak(self, message: str) -> bool:
        """쿨다운 조건을 만족할 때만 안내 문장을 음성 출력합니다."""
        if not self.enabled:
            return False
        now = time.time()

        if message == self.last_message and (now - self.last_spoken_at) < self.cooldown_sec:
            return False
        if (now - self.last_spoken_at) < self.cooldown_sec:
            return False

        spoken = self._speak_now(message)
        if spoken:
            self.last_message = message
            self.last_spoken_at = now
        return spoken

    def _speak_now(self, message: str) -> bool:
        if self._speak_with_gtts(message):
            return True

        if self._engine is not None:
            self._engine.say(message)
            self._engine.runAndWait()
            return True

        command_candidates = (
            ("spd-say", ["spd-say", "-l", "ko", message]),
            ("espeak", ["espeak", "-v", "ko", message]),
            ("say", ["say", "-v", "Yuna", message]),
            ("say", ["say", message]),
        )
        for command, args in command_candidates:
            if shutil.which(command):
                subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
        return False

    def _select_korean_voice(self) -> None:
        if self._engine is None:
            return
        try:
            voices = self._engine.getProperty("voices")
        except Exception:
            return

        for voice in voices:
            voice_text = " ".join(
                str(value).lower()
                for value in (
                    getattr(voice, "id", ""),
                    getattr(voice, "name", ""),
                    getattr(voice, "languages", ""),
                )
            )
            if "ko" in voice_text or "korean" in voice_text:
                self._engine.setProperty("voice", voice.id)
                return

    def _speak_with_gtts(self, message: str) -> bool:
        """gTTS로 한국어 MP3를 만들고 ffplay로 재생합니다."""
        if not shutil.which("ffplay"):
            return False
        try:
            from gtts import gTTS  # type: ignore
        except Exception:
            return False

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
                audio_path = handle.name
            gTTS(text=message, lang="ko").save(audio_path)
            process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            threading.Thread(
                target=self._cleanup_audio_after_playback,
                args=(process, audio_path),
                daemon=True,
            ).start()
            return True
        except Exception:
            return False

    @staticmethod
    def _cleanup_audio_after_playback(process: subprocess.Popen, audio_path: str) -> None:
        process.wait()
        try:
            os.unlink(audio_path)
        except OSError:
            pass
