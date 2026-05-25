from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.agents.audit import AuditEvolutionAgent
from src.agents.context import ContextAgent
from src.agents.decision import DecisionAgent
from src.agents.market_sensor import MarketSensorAgent
from src.agents.paper_execution import PaperExecutionAgent
from src.agents.vision_sensor import VisionSensorAgent
from src.config import Settings
from src.cooldown import CooldownTracker
from src.pipeline import PenaltyResearchPipeline, load_replay_events
from src.storage.jsonl_store import RunStore


DEFAULT_THRESHOLD_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DEFAULT_CONFIDENCE_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


@dataclass(frozen=True)
class TuneResult:
    probability_threshold: float
    min_confidence: float
    precision: float | None
    recall: float | None
    f1: float | None
    total_pnl: float
    paper_orders: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    evolution_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_threshold": self.probability_threshold,
            "min_confidence": self.min_confidence,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "total_pnl": self.total_pnl,
            "paper_orders": self.paper_orders,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "evolution_candidates": self.evolution_candidates,
        }


def run_grid_search(
    input_path: Path,
    thresholds: list[float] | None = None,
    confidences: list[float] | None = None,
    base_settings: Settings | None = None,
) -> list[TuneResult]:
    raw_events = load_replay_events(input_path)
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLD_GRID
    confidences = confidences if confidences is not None else DEFAULT_CONFIDENCE_GRID
    base = base_settings or Settings()

    tmp_dir = TemporaryDirectory()
    results: list[TuneResult] = []
    base_dict = {
        "decision": {
            "probability_threshold": None,
            "min_confidence": None,
            "model_version": base.model_version,
        },
        "paper": {
            "simulated_size_usd": base.simulated_size_usd,
            "max_loss_usd": base.max_loss_usd,
            "min_liquidity_usd": base.min_liquidity_usd,
        },
        "polymarket": {
            "gamma_api_base": base.gamma_api_base,
            "geoblock_url": base.geoblock_url,
            "timeout_sec": base.timeout_sec,
            "max_retries": base.max_retries,
        },
        "data": {
            "runs_dir": base.runs_dir,
        },
        "project_name": base.project_name,
        "mode": base.mode,
        "enable_real_trading": base.enable_real_trading,
    }
    for threshold in thresholds:
        for confidence in confidences:
            cfg = dict(base_dict)
            cfg["decision"] = dict(cfg["decision"])
            cfg["decision"]["probability_threshold"] = threshold
            cfg["decision"]["min_confidence"] = confidence
            settings = Settings.from_dict(cfg)
            cooldown = None
            if settings.cooldown_sec > 0:
                cooldown = CooldownTracker(cooldown_sec=settings.cooldown_sec)
            pipeline = PenaltyResearchPipeline(
                settings=settings,
                vision_sensor=VisionSensorAgent(),
                market_sensor=MarketSensorAgent(),
                context_agent=ContextAgent(),
                decision_agent=DecisionAgent(settings),
                paper_execution_agent=PaperExecutionAgent(settings, cooldown=cooldown),
                audit_agent=AuditEvolutionAgent(probability_threshold=settings.probability_threshold),
                store=RunStore(Path(tmp_dir.name)),
            )

            tp = fp = tn = fn = 0
            paper_orders = 0
            candidates = 0
            total_pnl = 0.0
            for raw in raw_events:
                out = pipeline.process_raw_event(raw)
                audit = out["audit"]
                pred = out["prediction"]
                actual = audit["actual_outcome"]
                if actual == "unknown":
                    if out["paper_order"] is not None:
                        paper_orders += 1
                    if out["evolution_candidate"] is not None:
                        candidates += 1
                    total_pnl += float(audit.get("pnl_simulated", 0))
                    continue

                predicted_positive = pred["penalty_probability"] >= threshold
                actual_positive = actual in {
                    "yes", "true", "penalty", "penalty_awarded",
                    "penalty_scored", "var_penalty_awarded",
                }
                if predicted_positive and actual_positive:
                    tp += 1
                elif predicted_positive and not actual_positive:
                    fp += 1
                elif not predicted_positive and actual_positive:
                    fn += 1
                else:
                    tn += 1
                if out["paper_order"] is not None:
                    paper_orders += 1
                if out["evolution_candidate"] is not None:
                    candidates += 1
                total_pnl += float(audit.get("pnl_simulated", 0))

            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = None
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)

            results.append(
                TuneResult(
                    probability_threshold=threshold,
                    min_confidence=confidence,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    total_pnl=round(total_pnl, 4),
                    paper_orders=paper_orders,
                    true_positive=tp,
                    false_positive=fp,
                    true_negative=tn,
                    false_negative=fn,
                    evolution_candidates=candidates,
                )
            )
    return results


def format_tuning_report(results: list[TuneResult], top_k: int = 10) -> str:
    lines = [
        "Penalty Monitor - 参数网格搜索报告",
        "=" * 50,
        f"搜索组合数: {len(results)}",
        "",
    ]

    # 按 F1 排序
    scored = [r for r in results if r.f1 is not None]
    by_f1 = sorted(scored, key=lambda x: (x.f1 or 0, x.total_pnl), reverse=True)[:top_k]
    lines.append("Top F1 组合:")
    lines.append("-" * 50)
    lines.append(
        f"{'threshold':>10} {'confidence':>10} {'precision':>10} {'recall':>8} {'f1':>8} {'pnl':>10} {'orders':>7}"
    )
    for r in by_f1:
        lines.append(
            f"{r.probability_threshold:>10.2f} {r.min_confidence:>10.2f} "
            f"{(r.precision or 0):>10.2%} {(r.recall or 0):>8.2%} {(r.f1 or 0):>8.4f} "
            f"{r.total_pnl:>10.2f} {r.paper_orders:>7d}"
        )

    lines.append("")
    lines.append("Top PnL 组合:")
    lines.append("-" * 50)
    by_pnl = sorted(results, key=lambda x: x.total_pnl, reverse=True)[:top_k]
    lines.append(
        f"{'threshold':>10} {'confidence':>10} {'precision':>10} {'recall':>8} {'f1':>8} {'pnl':>10} {'orders':>7}"
    )
    for r in by_pnl:
        lines.append(
            f"{r.probability_threshold:>10.2f} {r.min_confidence:>10.2f} "
            f"{(r.precision or 0):>10.2%} {(r.recall or 0):>8.2%} {(r.f1 or 0):>8.4f} "
            f"{r.total_pnl:>10.2f} {r.paper_orders:>7d}"
        )

    lines.append("")
    lines.append("Top 精确率组合 (precision >= 0.8):")
    lines.append("-" * 50)
    high_prec = [r for r in scored if (r.precision or 0) >= 0.8]
    by_prec = sorted(high_prec, key=lambda x: (x.precision or 0, x.recall or 0), reverse=True)[:top_k]
    if not by_prec:
        lines.append("(无)")
    else:
        lines.append(
            f"{'threshold':>10} {'confidence':>10} {'precision':>10} {'recall':>8} {'f1':>8} {'pnl':>10} {'orders':>7}"
        )
        for r in by_prec:
            lines.append(
                f"{r.probability_threshold:>10.2f} {r.min_confidence:>10.2f} "
                f"{(r.precision or 0):>10.2%} {(r.recall or 0):>8.2%} {(r.f1 or 0):>8.4f} "
                f"{r.total_pnl:>10.2f} {r.paper_orders:>7d}"
            )

    return "\n".join(lines)


def save_tuning_results(results: list[TuneResult], path: Path) -> None:
    payload = {
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
