from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.notifier import NotifyConfig


try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class Settings:
    project_name: str = "penalty_moniter"
    mode: str = "paper"
    enable_real_trading: bool = False
    probability_threshold: float = 0.75
    min_confidence: float = 0.55
    model_version: str = "rule-v0.1"
    weight_box_contact: float = 0.25
    weight_fall: float = 0.15
    weight_protest: float = 0.15
    weight_ref_earpiece: float = 0.20
    weight_ref_var_walk: float = 0.20
    weight_stoppage: float = 0.05
    commentary_weight: float = 0.15
    simulated_size_usd: float = 100.0
    max_loss_usd: float = 25.0
    min_liquidity_usd: float = 0.0
    cooldown_sec: float = 0.0
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    geoblock_url: str = "https://polymarket.com/api/geoblock"
    timeout_sec: int = 15
    max_retries: int = 2
    runs_dir: str = "data/runs"
    notify_config: NotifyConfig = field(default_factory=NotifyConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Settings":
        decision = payload.get("decision", {})
        paper = payload.get("paper", {})
        poly = payload.get("polymarket", {})
        data = payload.get("data", {})
        notifications = payload.get("notifications", {})
        return cls(
            project_name=str(payload.get("project_name") or cls.project_name),
            mode=str(payload.get("mode") or cls.mode),
            enable_real_trading=bool(payload.get("enable_real_trading", False)),
            probability_threshold=float(decision.get("probability_threshold", cls.probability_threshold)),
            min_confidence=float(decision.get("min_confidence", cls.min_confidence)),
            model_version=str(decision.get("model_version") or cls.model_version),
            weight_box_contact=float(decision.get("weight_box_contact", cls.weight_box_contact)),
            weight_fall=float(decision.get("weight_fall", cls.weight_fall)),
            weight_protest=float(decision.get("weight_protest", cls.weight_protest)),
            weight_ref_earpiece=float(decision.get("weight_ref_earpiece", cls.weight_ref_earpiece)),
            weight_ref_var_walk=float(decision.get("weight_ref_var_walk", cls.weight_ref_var_walk)),
            weight_stoppage=float(decision.get("weight_stoppage", cls.weight_stoppage)),
            commentary_weight=float(payload.get("commentary", {}).get("weight", cls.commentary_weight)),
            simulated_size_usd=float(paper.get("simulated_size_usd", cls.simulated_size_usd)),
            max_loss_usd=float(paper.get("max_loss_usd", cls.max_loss_usd)),
            min_liquidity_usd=float(paper.get("min_liquidity_usd", cls.min_liquidity_usd)),
            cooldown_sec=float(paper.get("cooldown_sec", cls.cooldown_sec)),
            gamma_api_base=str(poly.get("gamma_api_base") or cls.gamma_api_base),
            geoblock_url=str(poly.get("geoblock_url") or cls.geoblock_url),
            timeout_sec=int(poly.get("timeout_sec", cls.timeout_sec)),
            max_retries=int(poly.get("max_retries", cls.max_retries)),
            runs_dir=str(data.get("runs_dir") or cls.runs_dir),
            notify_config=NotifyConfig.from_dict(notifications),
        )

    @classmethod
    def from_file(cls, path: Path) -> "Settings":
        payload = _load_mapping(path)
        return cls.from_dict(payload)

    def new_run_dir(self, root: Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / self.runs_dir / stamp

    def assert_paper_only(self) -> None:
        if self.mode != "paper" or self.enable_real_trading:
            raise ValueError("当前 MVP 只支持 paper 模式，真钱交易已禁用。")


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    return value if isinstance(value, dict) else {}
