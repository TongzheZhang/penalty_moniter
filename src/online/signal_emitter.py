from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSignal:
    penalty_probability: float      # 模型原始输出
    smoothed_probability: float     # 平滑后概率
    confidence: float               # 模型置信度
    seconds_to_event: float | None  # 预测距点球还有多少秒
    model_version: str = "unknown"
    triggered: bool = False         # 是否触发信号


class SignalEmitter:
    """信号平滑与发射器。

    对原始模型输出进行平滑、去抖动、阈值判断，最终决定是否触发点球预警。
    """

    def __init__(
        self,
        probability_threshold: float = 0.75,
        min_confidence: float = 0.55,
        smoothing_window: int = 5,
        cooldown_sec: float = 60.0,
        rise_edge_min_delta: float = 0.3,
    ) -> None:
        self.probability_threshold = probability_threshold
        self.min_confidence = min_confidence
        self.smoothing_window = smoothing_window
        self.cooldown_sec = cooldown_sec
        self.rise_edge_min_delta = rise_edge_min_delta

        self._prob_history: deque[float] = deque(maxlen=smoothing_window)
        self._last_trigger_time: float | None = None
        self._prev_smoothed: float = 0.0

    def emit(self, result: Any, current_time: float) -> ModelSignal:
        """处理单次推理结果，返回平滑后的信号。

        Args:
            result: InferenceResult 对象
            current_time: 当前时间戳（秒）
        """
        raw_prob = getattr(result, "probability", 0.0)
        confidence = getattr(result, "confidence", 0.0)
        seconds_to = getattr(result, "seconds_to_event", None)
        model_version = getattr(result, "model_version", "unknown")

        # 滑动平均平滑
        self._prob_history.append(raw_prob)
        smoothed = sum(self._prob_history) / len(self._prob_history)

        # 上升沿检测：平滑概率从 < threshold-delta 跃升到 > threshold
        rise_edge = (
            self._prev_smoothed < self.probability_threshold - self.rise_edge_min_delta
            and smoothed >= self.probability_threshold
        )

        # 冷却期检查
        in_cooldown = False
        if self._last_trigger_time is not None:
            if current_time - self._last_trigger_time < self.cooldown_sec:
                in_cooldown = True

        triggered = (
            rise_edge
            and confidence >= self.min_confidence
            and not in_cooldown
        )

        if triggered:
            self._last_trigger_time = current_time

        self._prev_smoothed = smoothed

        return ModelSignal(
            penalty_probability=round(raw_prob, 4),
            smoothed_probability=round(smoothed, 4),
            confidence=round(confidence, 4),
            seconds_to_event=round(seconds_to, 2) if seconds_to is not None else None,
            model_version=model_version,
            triggered=triggered,
        )

    def reset(self) -> None:
        """重置状态（用于新比赛）。"""
        self._prob_history.clear()
        self._last_trigger_time = None
        self._prev_smoothed = 0.0
