from __future__ import annotations

import hashlib
from typing import Any

from src.models import AuditRecord, EvolutionCandidate, EvidenceEvent, PaperOrder, Prediction


POSITIVE_OUTCOMES = {
    "yes",
    "true",
    "penalty",
    "penalty_awarded",
    "penalty_scored",
    "var_penalty_awarded",
}


class AuditEvolutionAgent:
    """生成赛后审计记录和待人工审核的进化候选。"""

    def __init__(self, probability_threshold: float = 0.75) -> None:
        self.probability_threshold = probability_threshold

    def audit(
        self,
        event: EvidenceEvent,
        prediction: Prediction,
        paper_order: PaperOrder | None,
        actual_outcome: Any,
        price_after_30s: float | None,
        price_after_120s: float | None,
        execution_blocks: list[str] | None = None,
    ) -> AuditRecord:
        actual_label = normalize_outcome(actual_outcome)
        actual_positive = is_positive_outcome(actual_outcome)
        predicted_positive = prediction.penalty_probability >= self.probability_threshold

        failure_reason = ""
        if actual_label == "unknown":
            failure_reason = "outcome_unknown"
        elif predicted_positive and not actual_positive:
            failure_reason = "false_positive"
        elif not predicted_positive and actual_positive:
            failure_reason = "missed_opportunity"
        elif predicted_positive and actual_positive and paper_order is None:
            failure_reason = "execution_blocked:" + ",".join(execution_blocks or ["unknown"])

        pnl = calculate_paper_pnl(
            order=paper_order,
            actual_positive=actual_positive,
            price_after_30s=price_after_30s,
            price_after_120s=price_after_120s,
        )

        return AuditRecord(
            event_id=event.event_id,
            prediction=prediction.to_dict(),
            paper_order=paper_order.to_dict() if paper_order is not None else None,
            actual_outcome=actual_label,
            price_after_30s=price_after_30s,
            price_after_120s=price_after_120s,
            pnl_simulated=round(pnl, 4),
            failure_reason=failure_reason,
        )

    def maybe_create_candidate(self, event: EvidenceEvent, audit: AuditRecord) -> EvolutionCandidate | None:
        if audit.failure_reason not in {"false_positive", "missed_opportunity"}:
            return None

        digest = hashlib.sha1(f"{event.event_id}:{audit.failure_reason}".encode("utf-8")).hexdigest()[:10]
        if audit.failure_reason == "false_positive":
            proposed = (
                "收紧可执行条件：对接近阈值的信号簇，要求更强的禁区接触确认，"
                "或要求盘口/比赛上下文提供第二来源确认后，才创建纸面订单。"
            )
        else:
            proposed = (
                "复盘漏报样本：检查回放关键帧；当接触、抗议、暂停信号同时出现但没有 VAR 走向信号时，"
                "考虑增加低阈值观察提醒，而不是直接交易。"
            )

        return EvolutionCandidate(
            candidate_id=f"cand_{digest}",
            event_id=event.event_id,
            failure_reason=audit.failure_reason,
            proposed_change=proposed,
        )


def normalize_outcome(value: Any) -> str:
    if isinstance(value, bool):
        return "penalty_awarded" if value else "no_penalty"
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return text or "unknown"


def is_positive_outcome(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return normalize_outcome(value) in POSITIVE_OUTCOMES


def calculate_paper_pnl(
    order: PaperOrder | None,
    actual_positive: bool,
    price_after_30s: float | None,
    price_after_120s: float | None,
) -> float:
    if order is None:
        return 0.0
    exit_price = price_after_120s
    if exit_price is None:
        exit_price = price_after_30s
    if exit_price is None:
        exit_price = 1.0 if actual_positive else 0.0
    shares = order.simulated_size / order.reference_price
    return shares * (float(exit_price) - order.reference_price)
