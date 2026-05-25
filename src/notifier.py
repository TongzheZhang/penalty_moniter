from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class NotifyConfig:
    enabled: bool = True
    desktop: bool = True
    sound: bool = True
    min_probability: float = 0.75

    @classmethod
    def from_dict(cls, payload: dict | None) -> "NotifyConfig":
        p = payload or {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            desktop=bool(p.get("desktop", True)),
            sound=bool(p.get("sound", True)),
            min_probability=float(p.get("min_probability", 0.75)),
        )


class Notifier:
    """跨平台桌面通知 + 声音提醒。

    - Linux: notify-send
    - macOS: osascript
    - Windows: 暂无原生支持（终端响铃仍可用）
    """

    def __init__(self, config: NotifyConfig | None = None) -> None:
        self.config = config or NotifyConfig()
        self._system = platform.system()

    def maybe_notify(self, title: str, message: str, probability: float) -> None:
        if not self.config.enabled:
            return
        if probability < self.config.min_probability:
            return
        if self.config.desktop:
            self._desktop_notify(title, message)
        if self.config.sound:
            self._sound_alert()

    def _desktop_notify(self, title: str, message: str) -> None:
        try:
            if self._system == "Linux":
                subprocess.run(
                    ["notify-send", title, message, "--urgency=normal"],
                    check=False,
                    capture_output=True,
                )
            elif self._system == "Darwin":
                script = f'display notification "{message}" with title "{title}"'
                subprocess.run(
                    ["osascript", "-e", script],
                    check=False,
                    capture_output=True,
                )
        except Exception:
            pass

    @staticmethod
    def _sound_alert() -> None:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass
