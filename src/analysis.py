from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.audit import is_positive_outcome
from src.storage.jsonl_store import JsonlStore


def load_run_data(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "evidence": JsonlStore(run_dir / "evidence.jsonl").read_all(),
        "predictions": JsonlStore(run_dir / "predictions.jsonl").read_all(),
        "paper_orders": JsonlStore(run_dir / "paper_orders.jsonl").read_all(),
        "audit": JsonlStore(run_dir / "audit.jsonl").read_all(),
        "evolution_candidates": JsonlStore(run_dir / "evolution_candidates.jsonl").read_all(),
    }


def merge_evidence_prediction_audit(
    evidence: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_by_eid = {p["event_id"]: p for p in predictions}
    audit_by_eid = {a["event_id"]: a for a in audit}
    merged: list[dict[str, Any]] = []
    for ev in evidence:
        eid = ev["event_id"]
        merged.append({
            "evidence": ev,
            "prediction": pred_by_eid.get(eid),
            "audit": audit_by_eid.get(eid),
        })
    return merged


def _signal_fields() -> list[str]:
    return [
        "box_contact_score",
        "fall_score",
        "protest_score",
        "ref_earpiece_score",
        "ref_var_walk_score",
        "whistle_or_stoppage_score",
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_run(run_dir: Path) -> dict[str, Any]:
    data = load_run_data(run_dir)
    merged = merge_evidence_prediction_audit(data["evidence"], data["predictions"], data["audit"])

    total = len(merged)
    labeled = 0
    tp = fp = tn = fn = 0
    total_pnl = 0.0
    paper_orders = 0

    signal_buckets: dict[str, dict[str, list[float]]] = {
        field: {"tp": [], "fp": [], "tn": [], "fn": []}
        for field in _signal_fields()
    }
    reason_buckets: dict[str, int] = {}
    failure_buckets: dict[str, int] = {}

    threshold = 0.75
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            threshold = float(summary.get("probability_threshold", threshold))
        except Exception:
            pass

    for item in merged:
        ev = item["evidence"]
        pred = item["prediction"] or {}
        aud = item["audit"] or {}

        actual = aud.get("actual_outcome", "unknown")
        prob = pred.get("penalty_probability", 0.0)

        predicted_positive = prob >= threshold
        actual_positive = is_positive_outcome(actual)

        if actual != "unknown":
            labeled += 1
            if predicted_positive and actual_positive:
                bucket = "tp"
                tp += 1
            elif predicted_positive and not actual_positive:
                bucket = "fp"
                fp += 1
            elif not predicted_positive and actual_positive:
                bucket = "fn"
                fn += 1
            else:
                bucket = "tn"
                tn += 1

            signals = ev.get("signals", {})
            for field in _signal_fields():
                signal_buckets[field][bucket].append(float(signals.get(field, 0.0)))

        reasons = pred.get("reason_codes", [])
        for r in reasons:
            reason_buckets[r] = reason_buckets.get(r, 0) + 1

        failure = aud.get("failure_reason", "")
        if failure:
            failure_buckets[failure] = failure_buckets.get(failure, 0) + 1

        if aud.get("paper_order") is not None:
            paper_orders += 1
        total_pnl += float(aud.get("pnl_simulated", 0.0))

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    signal_analysis = {}
    for field in _signal_fields():
        signal_analysis[field] = {
            "tp_mean": round(_mean(signal_buckets[field]["tp"]), 4),
            "fp_mean": round(_mean(signal_buckets[field]["fp"]), 4),
            "fn_mean": round(_mean(signal_buckets[field]["fn"]), 4),
            "tn_mean": round(_mean(signal_buckets[field]["tn"]), 4),
        }

    return {
        "run_dir": str(run_dir),
        "total_events": total,
        "labeled_events": labeled,
        "paper_orders": paper_orders,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_pnl": round(total_pnl, 4),
        "signal_analysis": signal_analysis,
        "reason_distribution": reason_buckets,
        "failure_distribution": failure_buckets,
    }


def format_analysis_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "Penalty Monitor - 运行分析 report",
        "=" * 50,
    ]

    for r in results:
        lines.append("")
        lines.append(f"运行目录: {r['run_dir']}")
        lines.append("-" * 50)
        lines.append(f"事件总数       : {r['total_events']}")
        lines.append(f"有标签事件     : {r['labeled_events']}")
        lines.append(f"纸面订单       : {r['paper_orders']}")
        lines.append(f"TP/FP/TN/FN    : {r['true_positive']}/{r['false_positive']}/{r['true_negative']}/{r['false_negative']}")
        lines.append(f"精确率         : {r['precision']:.2%}" if r['precision'] is not None else "精确率         : -")
        lines.append(f"召回率         : {r['recall']:.2%}" if r['recall'] is not None else "召回率         : -")
        lines.append(f"F1             : {r['f1']:.4f}" if r['f1'] is not None else "F1             : -")
        lines.append(f"总纸面 PnL     : {r['total_pnl']:.4f}")

        lines.append("")
        lines.append("信号分布均值 (TP vs FP vs FN vs TN):")
        for field, vals in r["signal_analysis"].items():
            lines.append(
                f"  {field:30s} TP={vals['tp_mean']:.2f} FP={vals['fp_mean']:.2f} "
                f"FN={vals['fn_mean']:.2f} TN={vals['tn_mean']:.2f}"
            )

        if r["reason_distribution"]:
            lines.append("")
            lines.append("预测原因分布:")
            for reason, count in sorted(r["reason_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count}")

        if r["failure_distribution"]:
            lines.append("")
            lines.append("失败原因分布:")
            for failure, count in sorted(r["failure_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"  {failure}: {count}")

    return "\n".join(lines)


def save_analysis_report(results: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps({"runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
