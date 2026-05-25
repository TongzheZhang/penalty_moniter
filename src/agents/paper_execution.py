from __future__ import annotations

from src.config import Settings
from src.cooldown import CooldownTracker
from src.models import EvidenceEvent, PaperOrder, Prediction


class PaperExecutionAgent:
    """只在所有风控守门条件通过后创建纸面订单。"""

    def __init__(self, settings: Settings, cooldown: CooldownTracker | None = None) -> None:
        self.settings = settings
        self.cooldown = cooldown

    def maybe_create_order(self, event: EvidenceEvent, prediction: Prediction) -> tuple[PaperOrder | None, list[str]]:
        self.settings.assert_paper_only()

        blocked: list[str] = []
        snapshot = event.market_snapshot
        reference_price = snapshot.reference_price()

        if prediction.penalty_probability < self.settings.probability_threshold:
            blocked.append("probability_below_threshold")
        if prediction.confidence < self.settings.min_confidence:
            blocked.append("confidence_below_threshold")
        if not snapshot.is_mapped():
            blocked.append("market_or_token_unmapped")
        if reference_price is None or reference_price <= 0 or reference_price >= 1:
            blocked.append("invalid_reference_price")
        if snapshot.liquidity_usd is not None and snapshot.liquidity_usd < self.settings.min_liquidity_usd:
            blocked.append("liquidity_below_minimum")
        if self.cooldown is not None and self.cooldown.is_in_cooldown(event.match_id, event.timestamp_utc):
            blocked.append("cooldown_active")

        if blocked:
            return None, blocked

        max_notional = min(self.settings.simulated_size_usd, self.settings.max_loss_usd)
        order = PaperOrder(
            event_id=event.event_id,
            market_id=snapshot.market_id,
            token_id=snapshot.token_id,
            side="BUY_YES",
            reference_price=round(float(reference_price), 4),
            simulated_size=round(max_notional, 2),
            max_loss=round(max_notional, 2),
        )
        if self.cooldown is not None:
            self.cooldown.record(event.match_id, event.timestamp_utc)
        return order, []
