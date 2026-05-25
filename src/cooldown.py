from __future__ import annotations

from datetime import datetime, timezone


class CooldownTracker:
    """防止同一比赛在冷却期内重复创建纸面订单。"""

    def __init__(self, cooldown_sec: float = 30.0) -> None:
        self.cooldown_sec = cooldown_sec
        self._last_order_time: dict[str, datetime] = {}

    def is_in_cooldown(self, match_id: str, timestamp_utc: str) -> bool:
        """检查给定时间是否仍在冷却期内。"""
        now = _parse_iso(timestamp_utc)
        last = self._last_order_time.get(match_id)
        if last is None:
            return False
        return (now - last).total_seconds() < self.cooldown_sec

    def record(self, match_id: str, timestamp_utc: str) -> None:
        """记录一次成功创建订单的时间。"""
        self._last_order_time[match_id] = _parse_iso(timestamp_utc)

    def reset(self) -> None:
        """清空所有记录，用于测试或 replay 重新开始。"""
        self._last_order_time.clear()


def _parse_iso(value: str) -> datetime:
    """解析 ISO 格式时间字符串，兼容带 Z 后缀。"""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)
