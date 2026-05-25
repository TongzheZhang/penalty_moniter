from __future__ import annotations

from src.clients.polymarket_client import PolymarketClient
from src.models import MatchInfo


class MatchRegistryAgent:
    """维护研究阶段的比赛与市场映射。"""

    def __init__(self) -> None:
        self._matches: dict[str, MatchInfo] = {}

    def register(self, match: MatchInfo) -> None:
        self._matches[match.match_id] = match

    def get(self, match_id: str) -> MatchInfo | None:
        return self._matches.get(match_id)

    def all(self) -> list[MatchInfo]:
        return list(self._matches.values())

    def discover_from_polymarket(self, client: PolymarketClient, tag_id: int, limit: int = 100) -> list[MatchInfo]:
        matches = [client.to_match_info(event) for event in client.fetch_events(tag_id=tag_id, limit=limit)]
        for match in matches:
            self.register(match)
        return matches
