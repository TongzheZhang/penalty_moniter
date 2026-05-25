from __future__ import annotations

import time

from src.config import Settings
from src.models import EvidenceEvent, Prediction


class DecisionAgent:
    """第一版可解释规则模型，用于估算 VAR 点球概率。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def predict(self, event: EvidenceEvent) -> Prediction:
        started = time.perf_counter()
        signals = event.signals

        visual_score = (
            self.settings.weight_box_contact * signals.box_contact_score
            + self.settings.weight_fall * signals.fall_score
            + self.settings.weight_protest * signals.protest_score
            + self.settings.weight_ref_earpiece * signals.ref_earpiece_score
            + self.settings.weight_ref_var_walk * signals.ref_var_walk_score
            + self.settings.weight_stoppage * signals.whistle_or_stoppage_score
        )

        probability = 0.05 + 0.90 * visual_score
        reason_codes = self._reason_codes(event, visual_score)

        if event.match_context.var_history_penalty_rate is not None:
            referee_rate = event.match_context.var_history_penalty_rate
            probability += max(-0.05, min(0.05, (referee_rate - 0.50) * 0.10))
            reason_codes.append("referee_history_available")

        if event.match_context.has_score():
            goal_gap = abs((event.match_context.score_home or 0) - (event.match_context.score_away or 0))
            if goal_gap <= 1:
                reason_codes.append("high_leverage_scoreline")

        probability = round(max(0.0, min(0.99, probability)), 4)
        confidence = round(self._confidence(event), 4)
        expected_market = event.market_snapshot.market_id or "unmapped_penalty_or_match_winner_market"
        latency_ms = int((time.perf_counter() - started) * 1000)

        return Prediction(
            event_id=event.event_id,
            penalty_probability=probability,
            confidence=confidence,
            side=event.match_context.attacking_side or "unknown",
            expected_market=expected_market,
            reason_codes=reason_codes,
            model_version=self.settings.model_version,
            latency_ms=latency_ms,
        )

    def _confidence(self, event: EvidenceEvent) -> float:
        confidence = 0.25
        if event.frame_paths:
            confidence += 0.15
        if event.source and event.source != "unknown":
            confidence += 0.05
        if event.market_snapshot.is_mapped():
            confidence += 0.15
        if event.match_context.minute is not None:
            confidence += 0.10
        if event.match_context.has_score():
            confidence += 0.10
        if event.signals.max_score() >= 0.70:
            confidence += 0.20
        return max(0.0, min(0.95, confidence))

    @staticmethod
    def _reason_codes(event: EvidenceEvent, visual_score: float) -> list[str]:
        signals = event.signals
        reasons: list[str] = []
        if signals.box_contact_score >= 0.65:
            reasons.append("box_contact_high")
        if signals.fall_score >= 0.65:
            reasons.append("fall_high")
        if signals.protest_score >= 0.65:
            reasons.append("player_protest_high")
        if signals.ref_earpiece_score >= 0.65:
            reasons.append("ref_earpiece_high")
        if signals.ref_var_walk_score >= 0.65:
            reasons.append("ref_var_walk_high")
        if signals.whistle_or_stoppage_score >= 0.65:
            reasons.append("stoppage_high")
        if visual_score >= 0.80:
            reasons.append("multi_signal_cluster_strong")
        elif visual_score >= 0.65:
            reasons.append("multi_signal_cluster_medium")
        if not event.market_snapshot.is_mapped():
            reasons.append("market_unmapped")
        return reasons
