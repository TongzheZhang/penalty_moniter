from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SIGNAL_FIELDS = [
    "box_contact_score",
    "fall_score",
    "protest_score",
    "ref_earpiece_score",
    "ref_var_walk_score",
    "whistle_or_stoppage_score",
]


def load_or_create(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "signals": {},
        "market_snapshot": {},
        "match_context": {},
    }


def update_state(
    path: Path,
    signals: dict[str, float],
    market_snapshot: dict[str, Any] | None = None,
    match_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_or_create(path)

    for key, val in signals.items():
        payload["signals"][key] = round(max(0.0, min(1.0, float(val))), 4)

    if market_snapshot:
        payload["market_snapshot"].update(market_snapshot)
    if match_context:
        payload["match_context"].update(match_context)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
