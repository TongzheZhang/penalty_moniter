from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlStore:
    def __init__(self, path: Path, reset: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


class RunStore:
    def __init__(self, root: Path, reset: bool = True) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence = JsonlStore(self.root / "evidence.jsonl", reset=reset)
        self.predictions = JsonlStore(self.root / "predictions.jsonl", reset=reset)
        self.paper_orders = JsonlStore(self.root / "paper_orders.jsonl", reset=reset)
        self.audit = JsonlStore(self.root / "audit.jsonl", reset=reset)
        self.evolution_candidates = JsonlStore(self.root / "evolution_candidates.jsonl", reset=reset)

    def write_summary(self, payload: dict[str, Any]) -> None:
        (self.root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
