from __future__ import annotations

from datetime import datetime
from typing import Any


def optional_float(value: Any) -> float | None:
    """安全地将值转换为 float，无效时返回 None。"""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    """安全地将值转换为 int，无效时返回 None。"""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全地将值转换为 float，无效时返回默认值。"""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_score(value: Any) -> float:
    """将值限制在 [0.0, 1.0] 范围内。"""
    return max(0.0, min(1.0, safe_float(value)))


def parse_iso_timestamp(value: str) -> datetime:
    """解析 ISO 格式时间字符串，兼容带 Z 后缀。"""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def parse_iso_safe(value: Any) -> datetime | None:
    """安全地解析 ISO 时间字符串，无效时返回 None。"""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
