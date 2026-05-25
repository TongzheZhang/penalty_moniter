from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
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


@dataclass(frozen=True)
class BatchResult:
    variant_id: str
    config: dict[str, Any]
    summary: dict[str, Any]
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "config": self.config,
            "summary": self.summary,
            "run_dir": self.run_dir,
        }


def _set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = mapping
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


def generate_config_variants(
    base_path: Path,
    overrides: dict[str, list[Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """基于基础配置和字段覆盖列表生成所有配置变体。

    overrides 格式: {"decision.probability_threshold": [0.7, 0.8], "paper.cooldown_sec": [0, 30]}
    """
    base = _load_yaml_or_json(base_path)
    keys = list(overrides.keys())
    values_list = [overrides[k] for k in keys]

    variants: list[tuple[str, dict[str, Any]]] = []
    for combo in product(*values_list):
        cfg = deepcopy(base)
        label_parts: list[str] = []
        for key, val in zip(keys, combo):
            _set_nested(cfg, key, val)
            safe_val = str(val).replace(".", "_")
            label_parts.append(f"{key.split('.')[-1]}={safe_val}")
        variant_id = "_".join(label_parts) if label_parts else "base"
        variants.append((variant_id, cfg))
    return variants


def run_batch_replay(
    input_path: Path,
    base_config_path: Path,
    overrides: dict[str, list[Any]],
    output_root: Path,
) -> list[BatchResult]:
    raw_events = load_replay_events(input_path)
    variants = generate_config_variants(base_config_path, overrides)
    results: list[BatchResult] = []

    for variant_id, cfg_dict in variants:
        run_dir = output_root / variant_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # 将变体配置写入运行目录，方便追溯
        (run_dir / "variant_config.yaml").write_text(
            json.dumps(cfg_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        settings = Settings.from_dict(cfg_dict)
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
            store=RunStore(run_dir),
        )

        outputs = [pipeline.process_raw_event(raw) for raw in raw_events]
        from src.pipeline import summarize_outputs
        summary = summarize_outputs(outputs, threshold=settings.probability_threshold)
        summary["run_dir"] = str(run_dir)
        summary["variant_id"] = variant_id
        pipeline.store.write_summary(summary)

        results.append(BatchResult(variant_id=variant_id, config=cfg_dict, summary=summary, run_dir=str(run_dir)))

    return results


def format_batch_report(results: list[BatchResult]) -> str:
    lines = [
        "Penalty Monitor - 批量回测对比报告",
        "=" * 80,
        f"{'变体ID':<30} {'Events':>8} {'Orders':>8} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'PnL':>10}",
        "-" * 80,
    ]
    for r in results:
        s = r.summary
        prec = f"{s.get('precision'):.2%}" if s.get("precision") is not None else "-"
        rec = f"{s.get('recall'):.2%}" if s.get("recall") is not None else "-"
        lines.append(
            f"{r.variant_id:<30} {s.get('total_events', 0):>8} {s.get('paper_orders', 0):>8} "
            f"{s.get('true_positive', 0):>4} {s.get('false_positive', 0):>4} "
            f"{s.get('true_negative', 0):>4} {s.get('false_negative', 0):>4} "
            f"{prec:>10} {rec:>8} {s.get('total_pnl', 0):>10.4f}"
        )
    return "\n".join(lines)


def save_batch_results(results: list[BatchResult], path: Path) -> None:
    payload = {
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
