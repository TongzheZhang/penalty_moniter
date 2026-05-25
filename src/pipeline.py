from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.audit import AuditEvolutionAgent, is_positive_outcome, normalize_outcome
from src.agents.context import ContextAgent
from src.agents.decision import DecisionAgent
from src.agents.market_sensor import MarketSensorAgent
from src.agents.paper_execution import PaperExecutionAgent
from src.agents.vision_sensor import VisionSensorAgent
from src.config import Settings
from src.models import EvidenceEvent
from src.storage.jsonl_store import RunStore
from src.utils import optional_float


class PenaltyResearchPipeline:
    def __init__(
        self,
        settings: Settings,
        vision_sensor: VisionSensorAgent,
        market_sensor: MarketSensorAgent,
        context_agent: ContextAgent,
        decision_agent: DecisionAgent,
        paper_execution_agent: PaperExecutionAgent,
        audit_agent: AuditEvolutionAgent,
        store: RunStore,
    ) -> None:
        self.settings = settings
        self.vision_sensor = vision_sensor
        self.market_sensor = market_sensor
        self.context_agent = context_agent
        self.decision_agent = decision_agent
        self.paper_execution_agent = paper_execution_agent
        self.audit_agent = audit_agent
        self.store = store

    def process_raw_event(self, raw_event: dict[str, Any]) -> dict[str, Any]:
        event_payload = dict(raw_event)
        event_payload["signals"] = self.vision_sensor.extract(raw_event).to_dict()
        event_payload["market_snapshot"] = self.market_sensor.extract(raw_event).to_dict()
        event_payload["match_context"] = self.context_agent.extract(raw_event).to_dict()
        event = EvidenceEvent.from_dict(event_payload)

        prediction = self.decision_agent.predict(event)
        paper_order, execution_blocks = self.paper_execution_agent.maybe_create_order(event, prediction)
        audit = self.audit_agent.audit(
            event=event,
            prediction=prediction,
            paper_order=paper_order,
            actual_outcome=raw_event.get("actual_outcome"),
            price_after_30s=optional_float(raw_event.get("price_after_30s")),
            price_after_120s=optional_float(raw_event.get("price_after_120s")),
            execution_blocks=execution_blocks,
        )
        candidate = self.audit_agent.maybe_create_candidate(event, audit)

        self.store.evidence.append(event.to_dict())
        self.store.predictions.append(prediction.to_dict())
        if paper_order is not None:
            self.store.paper_orders.append(paper_order.to_dict())
        self.store.audit.append(audit.to_dict())
        if candidate is not None:
            self.store.evolution_candidates.append(candidate.to_dict())

        return {
            "event": event.to_dict(),
            "prediction": prediction.to_dict(),
            "paper_order": paper_order.to_dict() if paper_order is not None else None,
            "execution_blocks": execution_blocks,
            "audit": audit.to_dict(),
            "evolution_candidate": candidate.to_dict() if candidate is not None else None,
        }

    def run_replay(self, input_path: Path) -> dict[str, Any]:
        raw_events = load_replay_events(input_path)
        outputs = [self.process_raw_event(raw) for raw in raw_events]
        summary = summarize_outputs(outputs, threshold=self.settings.probability_threshold)
        summary["run_dir"] = str(self.store.root)
        summary["input_path"] = str(input_path)
        summary["probability_threshold"] = self.settings.probability_threshold
        summary["min_confidence"] = self.settings.min_confidence
        summary["cooldown_sec"] = self.settings.cooldown_sec
        summary["model_version"] = self.settings.model_version
        self.store.write_summary(summary)
        return summary


def load_replay_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return [item for item in payload["events"] if isinstance(item, dict)]
    raise ValueError("Replay input must be a JSON array, {events: [...]}, or JSONL")


def summarize_outputs(outputs: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    paper_orders = 0
    candidates = 0
    total_latency = 0
    total_pnl = 0.0
    labeled_events = 0
    for item in outputs:
        prediction = item["prediction"]
        audit = item["audit"]
        total_pnl += float(audit.get("pnl_simulated", 0.0))
        if normalize_outcome(audit["actual_outcome"]) == "unknown":
            total_latency += int(prediction.get("latency_ms", 0))
            if item["paper_order"] is not None:
                paper_orders += 1
            if item["evolution_candidate"] is not None:
                candidates += 1
            continue
        labeled_events += 1
        predicted_positive = prediction["penalty_probability"] >= threshold
        actual_positive = is_positive_outcome(audit["actual_outcome"])
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
        if item["paper_order"] is not None:
            paper_orders += 1
        if item["evolution_candidate"] is not None:
            candidates += 1
        total_latency += int(prediction.get("latency_ms", 0))

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "status": "ok",
        "total_events": len(outputs),
        "labeled_events": labeled_events,
        "paper_orders": paper_orders,
        "audit_records": len(outputs),
        "evolution_candidates": candidates,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "total_pnl": round(total_pnl, 4),
        "avg_latency_ms": round(total_latency / len(outputs), 2) if outputs else 0,
    }



