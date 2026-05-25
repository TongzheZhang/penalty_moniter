from __future__ import annotations

from typing import Any


def format_replay_summary(summary: dict[str, Any]) -> str:
    precision = _fmt_ratio(summary.get("precision"))
    recall = _fmt_ratio(summary.get("recall"))
    lines = [
        "Penalty Monitor - 回放摘要",
        "=" * 40,
        f"状态                 : {summary.get('status')}",
        f"事件总数             : {summary.get('total_events')}",
        f"有标签事件           : {summary.get('labeled_events')}",
        f"纸面订单             : {summary.get('paper_orders')}",
        f"审计记录             : {summary.get('audit_records')}",
        f"进化候选             : {summary.get('evolution_candidates')}",
        f"TP/FP/TN/FN          : {summary.get('true_positive')}/"
        f"{summary.get('false_positive')}/{summary.get('true_negative')}/{summary.get('false_negative')}",
        f"精确率               : {precision}",
        f"召回率               : {recall}",
        f"平均延迟(ms)         : {summary.get('avg_latency_ms')}",
        f"运行目录             : {summary.get('run_dir')}",
    ]
    return "\n".join(lines)


def _fmt_ratio(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2%}"
