from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pipeline import load_replay_events
from src.storage.jsonl_store import JsonlStore


def load_unlabeled_events(source: Path) -> list[dict[str, Any]]:
    """从 replay 文件或运行目录的 evidence.jsonl 中加载未标注事件。"""
    if source.is_dir():
        evidence = JsonlStore(source / "evidence.jsonl").read_all()
        audit = JsonlStore(source / "audit.jsonl").read_all()
        audit_by_eid = {a["event_id"]: a for a in audit}
        events = []
        for ev in evidence:
            eid = ev["event_id"]
            actual = audit_by_eid.get(eid, {}).get("actual_outcome", "unknown")
            if actual == "unknown":
                events.append(ev)
        return events
    else:
        raw_events = load_replay_events(source)
        return [ev for ev in raw_events if ev.get("actual_outcome", "unknown") == "unknown"]


def interactive_annotate(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """交互式 CLI 标注。"""
    annotated: list[dict[str, Any]] = []
    print(f"共 {len(events)} 个未标注事件，开始交互式标注...")
    print("提示: 输入 y=点球/yes, n=无点球/no, u=未知/unknown, s=跳过, q=退出并保存\n")

    for idx, ev in enumerate(events, 1):
        eid = ev.get("event_id", "")
        match_id = ev.get("match_id", "")
        signals = ev.get("signals", {})
        print(f"[{idx}/{len(events)}] 事件: {eid} | 比赛: {match_id}")
        print(f"  信号: box_contact={signals.get('box_contact_score', 0):.2f} "
              f"fall={signals.get('fall_score', 0):.2f} "
              f"protest={signals.get('protest_score', 0):.2f} "
              f"ref_earpiece={signals.get('ref_earpiece_score', 0):.2f} "
              f"ref_var_walk={signals.get('ref_var_walk_score', 0):.2f} "
              f"stoppage={signals.get('whistle_or_stoppage_score', 0):.2f}")
        print(f"  盘口: {ev.get('market_snapshot', {})}")
        print(f"  上下文: {ev.get('match_context', {})}")

        while True:
            choice = input("标注结果 [y/n/u/s/q]? ").strip().lower()
            if choice in ("q", "quit"):
                print("退出标注，保存已标注结果...")
                return annotated
            if choice in ("s", "skip"):
                break
            if choice in ("y", "yes", "1"):
                ev["actual_outcome"] = "penalty_awarded"
                annotated.append(ev)
                break
            if choice in ("n", "no", "0"):
                ev["actual_outcome"] = "no_penalty"
                annotated.append(ev)
                break
            if choice in ("u", "unknown"):
                ev["actual_outcome"] = "unknown"
                annotated.append(ev)
                break
            print("无效输入，请重新输入。")
        print("")

    print(f"标注完成，共标注 {len(annotated)} 个事件。")
    return annotated


def batch_annotate(events: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    """使用预定义的 event_id -> outcome 映射批量标注。"""
    annotated: list[dict[str, Any]] = []
    for ev in events:
        eid = ev.get("event_id", "")
        if eid in mapping:
            ev["actual_outcome"] = mapping[eid]
            annotated.append(ev)
    return annotated


def save_annotated_events(events: list[dict[str, Any]], path: Path) -> None:
    if path.suffix.lower() == ".jsonl":
        path.write_text(
            "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events) + "\n",
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
