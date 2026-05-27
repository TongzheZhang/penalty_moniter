from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PenaltyAnnotation:
    match_id: str
    penalty_id: str
    start_sec: float          # 犯规动作开始
    var_start_sec: float | None = None
    whistle_sec: float | None = None
    kick_sec: float | None = None
    confidence: str = "likely"   # confirmed | likely | uncertain
    notes: str = ""
    labeler: str = "anonymous"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "penalty_id": self.penalty_id,
            "start_sec": self.start_sec,
            "var_start_sec": self.var_start_sec,
            "whistle_sec": self.whistle_sec,
            "kick_sec": self.kick_sec,
            "confidence": self.confidence,
            "notes": self.notes,
            "labeler": self.labeler,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PenaltyAnnotation":
        return cls(
            match_id=str(payload.get("match_id", "")),
            penalty_id=str(payload.get("penalty_id", "")),
            start_sec=float(payload.get("start_sec", 0)),
            var_start_sec=payload.get("var_start_sec"),
            whistle_sec=payload.get("whistle_sec"),
            kick_sec=payload.get("kick_sec"),
            confidence=str(payload.get("confidence", "likely")),
            notes=str(payload.get("notes", "")),
            labeler=str(payload.get("labeler", "anonymous")),
            created_at=str(payload.get("created_at", "")),
        )


class AnnotationStore:
    """管理单场比赛的标注数据。"""

    def __init__(self, match_dir: Path) -> None:
        self.match_dir = Path(match_dir)
        self.annotations_path = self.match_dir / "annotations.json"
        self.annotations: list[PenaltyAnnotation] = []
        self._load()

    def _load(self) -> None:
        if self.annotations_path.exists():
            try:
                data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
                self.annotations = [PenaltyAnnotation.from_dict(a) for a in data.get("annotations", [])]
            except (json.JSONDecodeError, KeyError):
                self.annotations = []

    def save(self) -> None:
        payload = {
            "match_id": self.match_dir.name,
            "count": len(self.annotations),
            "annotations": [a.to_dict() for a in self.annotations],
        }
        self.annotations_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, annotation: PenaltyAnnotation) -> None:
        self.annotations.append(annotation)
        self.save()

    def remove(self, penalty_id: str) -> bool:
        orig = len(self.annotations)
        self.annotations = [a for a in self.annotations if a.penalty_id != penalty_id]
        if len(self.annotations) < orig:
            self.save()
            return True
        return False

    def list(self) -> list[PenaltyAnnotation]:
        return list(self.annotations)


def run_cli_annotator(
    match_dir: Path,
    transcript_path: Path | None = None,
    labeler: str = "anonymous",
) -> int:
    """CLI 快速标注工具。

    简单的交互式终端标注，适合批量处理。
    """
    video_path = None
    for ext in (".mp4", ".mkv", ".flv"):
        candidates = list(match_dir.glob(f"match{ext}"))
        if candidates:
            video_path = candidates[0]
            break

    if video_path is None:
        print(f"错误: 未找到视频文件: {match_dir}")
        return 1

    store = AnnotationStore(match_dir)
    transcripts: list[dict[str, Any]] = []
    if transcript_path and transcript_path.exists():
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                transcripts.append(json.loads(line))

    print(f"\n=== 点球标注工具 ===")
    print(f"比赛: {match_dir.name}")
    print(f"视频: {video_path}")
    print(f"已有标注: {len(store.list())} 个")
    print("\n快捷键: p=标记点球  v=VAR开始  w=吹哨  k=罚球  d=删除最后  s=保存退出  q=不保存退出")
    print("时间格式: 分钟:秒 或 纯秒数 (如 72:15 或 4335)\n")

    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue

        if user_input == "q":
            print("退出（未保存新更改）")
            return 0

        if user_input == "s":
            store.save()
            print(f"已保存 {len(store.list())} 个标注")
            return 0

        if user_input == "d":
            if store.annotations:
                removed = store.annotations.pop()
                print(f"已删除: {removed.penalty_id}")
            continue

        # 解析命令和时间
        parts = user_input.split()
        if not parts:
            continue

        cmd = parts[0]
        if len(parts) < 2:
            print("用法: <命令> <时间> [备注]")
            continue

        time_str = parts[1]
        note = " ".join(parts[2:]) if len(parts) > 2 else ""

        # 解析时间
        try:
            if ":" in time_str:
                mm, ss = time_str.split(":")
                sec = int(mm) * 60 + float(ss)
            else:
                sec = float(time_str)
        except ValueError:
            print(f"时间格式错误: {time_str}")
            continue

        penalty_id = f"{match_dir.name}_penalty_{len(store.list()) + 1:03d}"
        annotation = PenaltyAnnotation(
            match_id=match_dir.name,
            penalty_id=penalty_id,
            start_sec=sec,
            notes=note,
            labeler=labeler,
        )

        if cmd == "p":
            annotation.confidence = "confirmed"
        elif cmd == "v":
            annotation.var_start_sec = sec
            print(f"VAR开始标记: {time_str}")
            continue
        elif cmd == "w":
            annotation.whistle_sec = sec
            print(f"吹哨标记: {time_str}")
            continue
        elif cmd == "k":
            annotation.kick_sec = sec
            print(f"罚球标记: {time_str}")
            continue
        else:
            print(f"未知命令: {cmd}")
            continue

        store.add(annotation)
        print(f"已添加: {penalty_id} @ {time_str} 备注={note or '-'}")

    return 0
