from __future__ import annotations

from typing import Any

from src.models import MarketSnapshot


class MarketSensorAgent:
    """构造决策和纸面执行所需的盘口快照。"""

    def extract(self, raw_event: dict[str, Any]) -> MarketSnapshot:
        return MarketSnapshot.from_dict(raw_event.get("market_snapshot"))
