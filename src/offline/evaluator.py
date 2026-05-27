from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.offline.model import ModelConfig, PenaltyPredictor
from src.offline.trainer import PenaltyDataset, collate_fn


@dataclass
class EvalMetrics:
    auc_roc: float
    avg_precision: float
    f1_at_05: float
    early_detection_30s: float
    early_detection_60s: float
    early_detection_120s: float
    false_alarm_per_hour: float
    top3_precision: float
    top5_precision: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "auc_roc": round(self.auc_roc, 4),
            "avg_precision": round(self.avg_precision, 4),
            "f1_at_05": round(self.f1_at_05, 4),
            "early_detection_30s": round(self.early_detection_30s, 4),
            "early_detection_60s": round(self.early_detection_60s, 4),
            "early_detection_120s": round(self.early_detection_120s, 4),
            "false_alarm_per_hour": round(self.false_alarm_per_hour, 4),
            "top3_precision": round(self.top3_precision, 4),
            "top5_precision": round(self.top5_precision, 4),
        }


def _compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算 AUC-ROC（简化实现，不依赖 sklearn）。"""
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    # Mann-Whitney U statistic
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    ranks = np.argsort(np.argsort(np.concatenate([pos_scores, neg_scores])))
    pos_ranks = ranks[:n_pos]
    u = pos_ranks.sum() - n_pos * (n_pos - 1) / 2
    return u / (n_pos * n_neg)


def _compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算 Average Precision（简化实现）。"""
    order = np.argsort(-y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0

    cumsum = np.cumsum(y_true)
    precisions = cumsum / np.arange(1, len(y_true) + 1)
    recalls = cumsum / n_pos

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            continue
        ap += np.max(precisions[recalls >= t])
    return ap / 11.0


def _compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _compute_early_detection(
    samples: list[dict[str, Any]],
    probs: np.ndarray,
    seconds: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """计算在点球前N秒成功预警的比例。"""
    detected = 0
    total = 0
    for i, s in enumerate(samples):
        if s.get("label") != 1.0:
            continue
        total += 1
        if probs[i] >= threshold:
            sec_to = s.get("seconds_to_penalty")
            if sec_to is not None and sec_to <= seconds[i]:
                detected += 1
    return detected / total if total else 0.0


class Evaluator:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.predictor = PenaltyPredictor(config=ModelConfig())
        self.model = self.predictor._build_module().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self, dataset_dir: Path) -> EvalMetrics:
        test_ds = PenaltyDataset(dataset_dir / "test.jsonl")
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        all_probs: list[float] = []
        all_labels: list[float] = []
        all_seconds: list[float] = []
        all_samples: list[dict[str, Any]] = []

        with torch.no_grad():
            for visual, text, context, labels, seconds in test_loader:
                visual = visual.to(self.device)
                text = text.to(self.device)
                context = context.to(self.device)

                prob, time_pred = self.model(visual, text, context)
                probs = torch.sigmoid(prob).cpu().numpy()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())
                all_seconds.extend(time_pred.cpu().numpy().tolist())

        # 读取样本用于 early detection 计算
        with (dataset_dir / "test.jsonl").open("r", encoding="utf-8") as f:
            all_samples = [json.loads(line) for line in f if line.strip()]

        y_true = np.array(all_labels)
        y_score = np.array(all_probs)
        y_pred = (y_score >= 0.5).astype(int)

        # 计算指标
        auc = _compute_auc_roc(y_true, y_score)
        ap = _compute_ap(y_true, y_score)
        f1 = _compute_f1(y_true, y_pred)

        # Early detection
        ed_30 = _compute_early_detection(all_samples, y_score, np.array(all_seconds), 30)
        ed_60 = _compute_early_detection(all_samples, y_score, np.array(all_seconds), 60)
        ed_120 = _compute_early_detection(all_samples, y_score, np.array(all_seconds), 120)

        # False alarm rate（每小时误报）
        total_hours = len(all_samples) * 5 / 3600  # 假设每个样本代表5秒
        false_alarms = np.sum((y_true == 0) & (y_pred == 1))
        far = false_alarms / total_hours if total_hours > 0 else 0.0

        # Top-K precision
        order = np.argsort(-y_score)
        top3 = y_true[order[:3]].mean() if len(y_true) >= 3 else 0.0
        top5 = y_true[order[:5]].mean() if len(y_true) >= 5 else 0.0

        metrics = EvalMetrics(
            auc_roc=auc,
            avg_precision=ap,
            f1_at_05=f1,
            early_detection_30s=ed_30,
            early_detection_60s=ed_60,
            early_detection_120s=ed_120,
            false_alarm_per_hour=far,
            top3_precision=top3,
            top5_precision=top5,
        )

        # 保存结果
        metrics_path = self.model_path.parent / "metrics.json"
        metrics_path.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        return metrics
