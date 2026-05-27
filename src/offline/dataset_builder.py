from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetSample:
    match_id: str
    slice_idx: int
    start_sec: float
    end_sec: float
    slice_path: str
    frame_paths: list[str] = field(default_factory=list)
    transcript_text: str = ""
    label: float = 0.0           # 0.0 = 非点球, 1.0 = 点球
    seconds_to_penalty: float | None = None
    match_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "slice_idx": self.slice_idx,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "slice_path": self.slice_path,
            "frame_paths": self.frame_paths,
            "transcript_text": self.transcript_text,
            "label": self.label,
            "seconds_to_penalty": self.seconds_to_penalty,
            "match_context": self.match_context,
        }


class DatasetBuilder:
    """从已标注的比赛构建训练数据集。"""

    def __init__(
        self,
        matches_dir: Path,
        slice_window: float = 10.0,
        slice_stride: float = 5.0,
        pos_buffer_sec: float = 60.0,
        neg_ratio: float = 4.0,
        random_seed: int = 42,
    ) -> None:
        self.matches_dir = Path(matches_dir)
        self.slice_window = slice_window
        self.slice_stride = slice_stride
        self.pos_buffer_sec = pos_buffer_sec
        self.neg_ratio = neg_ratio
        self.rng = random.Random(random_seed)

    def _load_annotations(self, match_dir: Path) -> list[dict[str, Any]]:
        path = match_dir / "annotations.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("annotations", [])
        except (json.JSONDecodeError, KeyError):
            return []

    def _load_transcript(self, match_dir: Path) -> list[dict[str, Any]]:
        path = match_dir / "transcript.jsonl"
        if not path.exists():
            return []
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except json.JSONDecodeError:
            return []

    def _find_transcript_in_range(
        self,
        transcripts: list[dict[str, Any]],
        start_sec: float,
        end_sec: float,
    ) -> str:
        texts: list[str] = []
        for seg in transcripts:
            seg_start = seg.get("start_sec", seg.get("start", 0))
            seg_end = seg.get("end_sec", seg.get("end", 0))
            if seg_end >= start_sec and seg_start <= end_sec:
                texts.append(seg.get("text", ""))
        return " ".join(texts)

    def _find_slices_for_match(self, match_dir: Path) -> list[tuple[int, float, float, Path]]:
        slices_dir = match_dir / "slices"
        if not slices_dir.exists():
            return []
        results: list[tuple[int, float, float, Path]] = []
        for p in sorted(slices_dir.glob("slice_*.mp4")):
            # 从文件名解析起止时间: slice_{idx:05d}_{start}s_{end}s.mp4
            try:
                parts = p.stem.split("_")
                idx = int(parts[1])
                start = float(parts[2].replace("s", ""))
                end = float(parts[3].replace("s", ""))
                results.append((idx, start, end, p))
            except (IndexError, ValueError):
                continue
        return results

    def _find_keyframes(self, match_dir: Path, slice_stem: str) -> list[str]:
        kf_dir = match_dir / "features" / "keyframes" / slice_stem
        if not kf_dir.exists():
            return []
        return [str(p) for p in sorted(kf_dir.glob("frame_*.jpg"))]

    def build(self, output_dir: Path) -> dict[str, Any]:
        """构建数据集并保存为 train/val/test.jsonl。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        positive_samples: list[DatasetSample] = []
        negative_samples: list[DatasetSample] = []

        for match_dir in sorted(self.matches_dir.iterdir()):
            if not match_dir.is_dir():
                continue

            annotations = self._load_annotations(match_dir)
            transcripts = self._load_transcript(match_dir)
            slices = self._find_slices_for_match(match_dir)

            if not slices:
                continue

            # 计算正样本时间范围（点球前后 buffer）
            pos_ranges: list[tuple[float, float]] = []
            for ann in annotations:
                start = float(ann.get("start_sec", 0))
                pos_ranges.append((start - self.pos_buffer_sec, start + self.pos_buffer_sec))

            for idx, sl_start, sl_end, sl_path in slices:
                text = self._find_transcript_in_range(transcripts, sl_start, sl_end)
                kframes = self._find_keyframes(match_dir, sl_path.stem)

                sample = DatasetSample(
                    match_id=match_dir.name,
                    slice_idx=idx,
                    start_sec=sl_start,
                    end_sec=sl_end,
                    slice_path=str(sl_path),
                    frame_paths=kframes,
                    transcript_text=text,
                    match_context={},
                )

                # 判断是正样本还是负样本
                is_positive = False
                min_dist: float | None = None
                for r_start, r_end in pos_ranges:
                    if sl_start <= r_end and sl_end >= r_start:
                        is_positive = True
                        # 计算距离最近点球的时间
                        for ann in annotations:
                            ann_start = float(ann.get("start_sec", 0))
                            dist = abs((sl_start + sl_end) / 2 - ann_start)
                            if min_dist is None or dist < min_dist:
                                min_dist = dist
                        break

                if is_positive:
                    sample.label = 1.0
                    sample.seconds_to_penalty = min_dist
                    positive_samples.append(sample)
                else:
                    negative_samples.append(sample)

        # 采样负样本以控制比例
        target_neg = int(len(positive_samples) * self.neg_ratio)
        if len(negative_samples) > target_neg:
            self.rng.shuffle(negative_samples)
            negative_samples = negative_samples[:target_neg]

        all_samples = positive_samples + negative_samples
        self.rng.shuffle(all_samples)

        # 划分 train/val/test (70/15/15)
        n = len(all_samples)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train = all_samples[:n_train]
        val = all_samples[n_train : n_train + n_val]
        test = all_samples[n_train + n_val :]

        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            path = output_dir / f"{split_name}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for s in split_data:
                    f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

        stats = {
            "total": n,
            "positive": len(positive_samples),
            "negative": len(negative_samples),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "pos_ratio": round(len(positive_samples) / n, 4) if n else 0,
        }

        stats_path = output_dir / "stats.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

        return stats
