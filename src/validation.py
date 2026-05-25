from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.agents.audit import normalize_outcome

VALID_OUTCOMES = {
    "yes", "true", "penalty", "penalty_awarded", "penalty_scored",
    "var_penalty_awarded", "no_penalty", "false", "no", "unknown",
}

SIGNAL_FIELDS = [
    "box_contact_score", "fall_score", "protest_score",
    "ref_earpiece_score", "ref_var_walk_score", "whistle_or_stoppage_score",
]


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # error | warning
    event_id: str
    field: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "event_id": self.event_id,
            "field": self.field,
            "message": self.message,
        }


def validate_events(events: list[dict[str, Any]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    match_events: dict[str, list[tuple[str, str]]] = {}

    for idx, ev in enumerate(events):
        eid = ev.get("event_id", "")
        # event_id
        if not eid:
            issues.append(ValidationIssue("error", f"<index:{idx}>", "event_id", "event_id 缺失或为空"))
            eid = f"<index:{idx}>"
        elif eid in seen_ids:
            issues.append(ValidationIssue("error", eid, "event_id", f"event_id 重复: {eid}"))
        else:
            seen_ids.add(eid)

        # match_id
        if not ev.get("match_id"):
            issues.append(ValidationIssue("error", eid, "match_id", "match_id 缺失或为空"))

        # timestamp_utc
        ts = ev.get("timestamp_utc")
        if ts:
            parsed_ts = _parse_iso_safe(ts)
            if parsed_ts is None:
                issues.append(ValidationIssue("error", eid, "timestamp_utc", f"时间格式无效: {ts}"))
            else:
                match_id = str(ev.get("match_id", ""))
                match_events.setdefault(match_id, []).append((eid, str(ts)))
        else:
            issues.append(ValidationIssue("warning", eid, "timestamp_utc", "timestamp_utc 缺失"))

        # signals
        signals = ev.get("signals")
        if isinstance(signals, dict):
            for field in SIGNAL_FIELDS:
                val = signals.get(field)
                if val is not None:
                    try:
                        fval = float(val)
                        if not 0.0 <= fval <= 1.0:
                            issues.append(ValidationIssue("error", eid, f"signals.{field}", f"值 {fval} 超出 [0,1] 范围"))
                    except (TypeError, ValueError):
                        issues.append(ValidationIssue("error", eid, f"signals.{field}", f"非数字值: {val}"))
        else:
            issues.append(ValidationIssue("warning", eid, "signals", "signals 缺失或不是对象"))

        # market_snapshot prices
        snapshot = ev.get("market_snapshot")
        if isinstance(snapshot, dict):
            for price_field in ("best_bid", "best_ask", "last_price"):
                val = snapshot.get(price_field)
                if val is not None and val != "":
                    try:
                        fval = float(val)
                        if not 0.0 < fval < 1.0:
                            issues.append(ValidationIssue("warning", eid, f"market_snapshot.{price_field}", f"价格 {fval} 不在 (0,1) 区间"))
                    except (TypeError, ValueError):
                        issues.append(ValidationIssue("error", eid, f"market_snapshot.{price_field}", f"非数字值: {val}"))
        else:
            issues.append(ValidationIssue("warning", eid, "market_snapshot", "market_snapshot 缺失或不是对象"))

        # actual_outcome
        outcome = ev.get("actual_outcome")
        if outcome is not None and outcome != "":
            if normalize_outcome(outcome) not in VALID_OUTCOMES:
                issues.append(ValidationIssue("warning", eid, "actual_outcome", f"未知结果值: {outcome}"))

        # match_context
        ctx = ev.get("match_context")
        if isinstance(ctx, dict):
            minute = ctx.get("minute")
            if minute is not None:
                try:
                    m = int(minute)
                    if not 0 <= m <= 120:
                        issues.append(ValidationIssue("warning", eid, "match_context.minute", f"比赛分钟 {m} 不在 0-120 范围"))
                except (TypeError, ValueError):
                    issues.append(ValidationIssue("error", eid, "match_context.minute", f"非整数值: {minute}"))
            side = ctx.get("attacking_side")
            if side is not None and str(side).lower() not in {"home", "away", "unknown"}:
                issues.append(ValidationIssue("warning", eid, "match_context.attacking_side", f"未知值: {side}"))
        else:
            issues.append(ValidationIssue("warning", eid, "match_context", "match_context 缺失或不是对象"))

        # frame_paths
        frames = ev.get("frame_paths")
        if frames is not None and not isinstance(frames, list):
            issues.append(ValidationIssue("error", eid, "frame_paths", "frame_paths 应为数组"))

    # 时间顺序检查
    for match_id, event_list in match_events.items():
        if len(event_list) < 2:
            continue
        parsed = []
        for eid, ts in event_list:
            dt = _parse_iso_safe(ts)
            if dt is not None:
                parsed.append((eid, dt))
        parsed.sort(key=lambda x: x[1])
        sorted_ids = [eid for eid, _ in parsed]
        original_ids = [eid for eid, _ in event_list]
        if sorted_ids != original_ids:
            issues.append(ValidationIssue(
                "warning", match_id, "timestamp_utc",
                f"比赛 {match_id} 的事件未按时间顺序排列"
            ))

    return issues


def format_validation_report(issues: list[ValidationIssue]) -> str:
    if not issues:
        return "✅ 数据验证通过，未发现任何问题。"

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    lines = [
        "Penalty Monitor - 数据验证报告",
        "=" * 50,
        f"错误: {len(errors)} | 警告: {len(warnings)}",
        "",
    ]

    if errors:
        lines.append("❌ 错误:")
        for issue in errors:
            lines.append(f"  [{issue.event_id}] {issue.field}: {issue.message}")
        lines.append("")

    if warnings:
        lines.append("⚠️  警告:")
        for issue in warnings:
            lines.append(f"  [{issue.event_id}] {issue.field}: {issue.message}")

    return "\n".join(lines)


def _parse_iso_safe(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
