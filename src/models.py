from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.time_utils import parse_utc, utc_now_iso
from src.utils import clamp_score, optional_float, optional_int, safe_float


@dataclass(frozen=True)
class SignalScores:
    box_contact_score: float = 0.0
    fall_score: float = 0.0
    protest_score: float = 0.0
    ref_earpiece_score: float = 0.0
    ref_var_walk_score: float = 0.0
    whistle_or_stoppage_score: float = 0.0
    commentary_score: float = 0.0
    commentary_triggered: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SignalScores":
        payload = payload or {}
        return cls(
            box_contact_score=clamp_score(payload.get("box_contact_score")),
            fall_score=clamp_score(payload.get("fall_score")),
            protest_score=clamp_score(payload.get("protest_score")),
            ref_earpiece_score=clamp_score(payload.get("ref_earpiece_score")),
            ref_var_walk_score=clamp_score(payload.get("ref_var_walk_score")),
            whistle_or_stoppage_score=clamp_score(payload.get("whistle_or_stoppage_score")),
            commentary_score=clamp_score(payload.get("commentary_score")),
            commentary_triggered=bool(payload.get("commentary_triggered", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def max_score(self) -> float:
        return max(
            self.box_contact_score,
            self.fall_score,
            self.protest_score,
            self.ref_earpiece_score,
            self.ref_var_walk_score,
            self.whistle_or_stoppage_score,
            self.commentary_score,
        )


@dataclass(frozen=True)
class MarketSnapshot:
    market_id: str = ""
    token_id: str = ""
    best_bid: float | None = None
    best_ask: float | None = None
    last_price: float | None = None
    liquidity_usd: float | None = None
    timestamp_utc: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MarketSnapshot":
        payload = payload or {}
        return cls(
            market_id=str(payload.get("market_id") or ""),
            token_id=str(payload.get("token_id") or ""),
            best_bid=optional_float(payload.get("best_bid")),
            best_ask=optional_float(payload.get("best_ask")),
            last_price=optional_float(payload.get("last_price")),
            liquidity_usd=optional_float(payload.get("liquidity_usd")),
            timestamp_utc=parse_utc(payload.get("timestamp_utc")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def reference_price(self) -> float | None:
        if self.best_ask is not None and self.best_ask > 0:
            return self.best_ask
        if self.last_price is not None and self.last_price > 0:
            return self.last_price
        if self.best_bid is not None and self.best_bid > 0:
            return self.best_bid
        return None

    def is_mapped(self) -> bool:
        return bool(self.market_id and self.token_id)


@dataclass(frozen=True)
class MatchContext:
    home: str = ""
    away: str = ""
    score_home: int | None = None
    score_away: int | None = None
    minute: int | None = None
    red_cards_home: int = 0
    red_cards_away: int = 0
    referee: str = ""
    var_history_penalty_rate: float | None = None
    attacking_side: str = "unknown"
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MatchContext":
        payload = payload or {}
        return cls(
            home=str(payload.get("home") or ""),
            away=str(payload.get("away") or ""),
            score_home=optional_int(payload.get("score_home")),
            score_away=optional_int(payload.get("score_away")),
            minute=optional_int(payload.get("minute")),
            red_cards_home=int(safe_float(payload.get("red_cards_home"), 0)),
            red_cards_away=int(safe_float(payload.get("red_cards_away"), 0)),
            referee=str(payload.get("referee") or ""),
            var_history_penalty_rate=optional_float(payload.get("var_history_penalty_rate")),
            attacking_side=str(payload.get("attacking_side") or "unknown"),
            notes=str(payload.get("notes") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has_score(self) -> bool:
        return self.score_home is not None and self.score_away is not None


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    match_id: str
    timestamp_utc: str
    video_ts: float | None
    source: str
    frame_paths: list[str]
    signals: SignalScores
    market_snapshot: MarketSnapshot
    match_context: MatchContext

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceEvent":
        return cls(
            event_id=str(payload.get("event_id") or ""),
            match_id=str(payload.get("match_id") or ""),
            timestamp_utc=parse_utc(payload.get("timestamp_utc")),
            video_ts=optional_float(payload.get("video_ts")),
            source=str(payload.get("source") or "unknown"),
            frame_paths=[str(item) for item in payload.get("frame_paths", [])],
            signals=SignalScores.from_dict(payload.get("signals")),
            market_snapshot=MarketSnapshot.from_dict(payload.get("market_snapshot")),
            match_context=MatchContext.from_dict(payload.get("match_context")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "match_id": self.match_id,
            "timestamp_utc": self.timestamp_utc,
            "video_ts": self.video_ts,
            "source": self.source,
            "frame_paths": list(self.frame_paths),
            "signals": self.signals.to_dict(),
            "market_snapshot": self.market_snapshot.to_dict(),
            "match_context": self.match_context.to_dict(),
        }


@dataclass(frozen=True)
class Prediction:
    event_id: str
    penalty_probability: float
    confidence: float
    side: str
    expected_market: str
    reason_codes: list[str]
    model_version: str
    latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperOrder:
    event_id: str
    market_id: str
    token_id: str
    side: str
    reference_price: float
    simulated_size: float
    max_loss: float
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditRecord:
    event_id: str
    prediction: dict[str, Any]
    paper_order: dict[str, Any] | None
    actual_outcome: str
    price_after_30s: float | None
    price_after_120s: float | None
    pnl_simulated: float
    failure_reason: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvolutionCandidate:
    candidate_id: str
    event_id: str
    failure_reason: str
    proposed_change: str
    status: str = "pending_human_review"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchInfo:
    match_id: str
    title: str
    slug: str
    start_time: str = ""
    market_id: str = ""
    token_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommentaryTranscript:
    timestamp_utc: str = field(default_factory=utc_now_iso)
    video_ts: float | None = None
    text: str = ""
    keywords_hit: list[str] = field(default_factory=list)
    urgency_score: float = 0.0
    raw_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoClip:
    clip_path: str
    start_ts: float
    end_ts: float
    trigger_event_id: str
    trigger_probability: float
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

