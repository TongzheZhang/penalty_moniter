from __future__ import annotations

from typing import Any

from src.models import MatchContext


class ContextAgent:
    """归一化比赛上下文；字段缺失时降级为空特征，不阻塞主流程。"""

    def extract(self, raw_event: dict[str, Any]) -> MatchContext:
        return MatchContext.from_dict(raw_event.get("match_context"))
