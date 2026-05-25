from __future__ import annotations

import json
import time
from typing import Any

import requests

from src.models import MatchInfo


class PolymarketClientError(RuntimeError):
    pass


class PolymarketClient:
    """用于研究数据采集的 Polymarket 只读客户端。"""

    def __init__(
        self,
        gamma_api_base: str,
        geoblock_url: str,
        timeout_sec: int = 15,
        max_retries: int = 2,
    ) -> None:
        self.gamma_api_base = gamma_api_base.rstrip("/")
        self.geoblock_url = geoblock_url
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def fetch_events(self, tag_id: int, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._get_json(
            f"{self.gamma_api_base}/events",
            params={"active": "true", "closed": "false", "tag_id": tag_id, "limit": limit},
        )
        if not isinstance(payload, list):
            raise PolymarketClientError("Gamma events 返回值不是列表")
        return payload

    def check_geoblock(self) -> dict[str, Any]:
        payload = self._get_json(self.geoblock_url, params=None)
        if not isinstance(payload, dict):
            raise PolymarketClientError("Geoblock 返回值不是对象")
        return payload

    def to_match_info(self, event: dict[str, Any]) -> MatchInfo:
        research = self.to_research_market(event)
        return MatchInfo(
            match_id=research["match_id"],
            title=research["title"],
            slug=research["slug"],
            start_time=research["start_time"],
            market_id=research["market_id"],
            token_id=research["token_id"],
        )

    def to_research_market(self, event: dict[str, Any]) -> dict[str, Any]:
        markets = event.get("markets") or []
        first_market = markets[0] if markets and isinstance(markets[0], dict) else {}
        token_ids = _parse_json_list(first_market.get("clobTokenIds"))
        prices = _parse_json_list(first_market.get("outcomePrices"))
        market_id = str(first_market.get("conditionId") or first_market.get("id") or event.get("id") or "")
        token_id = str(token_ids[0]) if token_ids else ""
        price = _safe_float(prices[0]) if prices else _safe_float(first_market.get("lastTradePrice"))
        return {
            "match_id": str(event.get("id") or event.get("slug") or ""),
            "title": str(event.get("title") or ""),
            "slug": str(event.get("slug") or ""),
            "start_time": str(event.get("startDate") or event.get("startTime") or ""),
            "market_id": market_id,
            "token_id": token_id,
            "last_price": price,
            "volume": _safe_float(event.get("volume")),
            "liquidity": _safe_float(event.get("liquidity")),
            "closed": bool(event.get("closed", False)),
            "active": bool(event.get("active", False)),
        }

    def _get_json(self, url: str, params: dict[str, Any] | None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=self.timeout_sec)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.25 * (2**attempt))
        raise PolymarketClientError(str(last_error))


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
