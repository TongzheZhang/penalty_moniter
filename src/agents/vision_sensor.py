from __future__ import annotations

from typing import Any

from src.models import SignalScores


class VisionSensorAgent:
    """视觉证据适配器。

    MVP 阶段消费预计算或人工标注的特征分数。后续接入真实视频模型时，
    只需要替换这个类，后续决策和审计接口可以保持不变。
    """

    def extract(self, raw_event: dict[str, Any]) -> SignalScores:
        return SignalScores.from_dict(raw_event.get("signals"))
