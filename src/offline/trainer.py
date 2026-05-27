from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.offline.model import ModelConfig, PenaltyPredictor


@dataclass
class TrainConfig:
    dataset_dir: str = "data/datasets/v1"
    checkpoint_dir: str = "data/models"
    model_name: str = "penalty_predictor_v1"
    device: str = "cpu"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    early_stop_patience: int = 10
    label_smoothing: float = 0.1
    pos_weight: float = 3.0
    aux_loss_weight: float = 0.2
    export_onnx: bool = True


class PenaltyDataset(Dataset):
    """加载 DatasetBuilder 生成的 jsonl 文件。"""

    def __init__(self, jsonl_path: Path, seq_len: int = 6) -> None:
        self.samples: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        # 视觉特征：如果没有预计算，用随机初始化占位（实际应从 .pt 加载）
        n_frames = len(s.get("frame_paths", []))
        visual = torch.randn(self.seq_len, 512) * 0.1  # 占位

        # 文本特征：占位（实际应从 BERT 编码加载）
        text = torch.randn(self.seq_len, 768) * 0.1

        # 上下文：比分、分钟等
        ctx = s.get("match_context", {})
        context = torch.tensor([
            float(ctx.get("minute", 0)) / 120.0,
            float(ctx.get("score_home", 0)) / 10.0,
            float(ctx.get("score_away", 0)) / 10.0,
            float(ctx.get("red_cards_home", 0)) / 5.0,
            float(ctx.get("red_cards_away", 0)) / 5.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=torch.float32)

        label = torch.tensor(float(s.get("label", 0.0)), dtype=torch.float32)
        seconds = torch.tensor(float(s.get("seconds_to_penalty", 0.0) or 0.0), dtype=torch.float32)

        return visual, text, context, label, seconds


def collate_fn(batch: list[Any]) -> tuple[torch.Tensor, ...]:
    visuals, texts, contexts, labels, seconds = zip(*batch)
    return (
        torch.stack(visuals),
        torch.stack(texts),
        torch.stack(contexts),
        torch.stack(labels),
        torch.stack(seconds),
    )


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict[str, Any]:
        dataset_dir = Path(self.config.dataset_dir)
        train_ds = PenaltyDataset(dataset_dir / "train.jsonl")
        val_ds = PenaltyDataset(dataset_dir / "val.jsonl")

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model_cfg = ModelConfig(pos_weight=self.config.pos_weight)
        predictor = PenaltyPredictor(config=model_cfg)
        model = predictor._build_module().to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        pos_weight = torch.tensor([self.config.pos_weight], device=self.device)
        criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion_reg = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self.config.num_epochs + 1):
            model.train()
            train_loss = 0.0
            for visual, text, context, labels, seconds in train_loader:
                visual = visual.to(self.device)
                text = text.to(self.device)
                context = context.to(self.device)
                labels = labels.to(self.device)
                seconds = seconds.to(self.device)

                optimizer.zero_grad()
                prob, time_pred = model(visual, text, context)

                loss_cls = criterion_cls(prob, labels)
                loss_reg = criterion_reg(time_pred, seconds)
                loss = loss_cls + self.config.aux_loss_weight * loss_reg

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            all_probs: list[float] = []
            all_labels: list[float] = []
            with torch.no_grad():
                for visual, text, context, labels, seconds in val_loader:
                    visual = visual.to(self.device)
                    text = text.to(self.device)
                    context = context.to(self.device)
                    labels = labels.to(self.device)
                    seconds = seconds.to(self.device)

                    prob, time_pred = model(visual, text, context)
                    loss_cls = criterion_cls(prob, labels)
                    loss_reg = criterion_reg(time_pred, seconds)
                    loss = loss_cls + self.config.aux_loss_weight * loss_reg
                    val_loss += loss.item()

                    all_probs.extend(torch.sigmoid(prob).cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            val_loss /= len(val_loader)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
            })

            print(f"Epoch {epoch}/{self.config.num_epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = self.checkpoint_dir / "best.pt"
                torch.save(model.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Save final
        final_path = self.checkpoint_dir / "final.pt"
        torch.save(model.state_dict(), final_path)

        # Training log
        log_path = self.checkpoint_dir / "training_log.json"
        log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        # Export ONNX
        if self.config.export_onnx:
            dummy_v = torch.randn(1, 6, 512).to(self.device)
            dummy_t = torch.randn(1, 6, 768).to(self.device)
            dummy_c = torch.randn(1, 16).to(self.device)
            onnx_path = self.checkpoint_dir / "model.onnx"
            predictor.to_onnx(onnx_path, dummy_v, dummy_t, dummy_c)
            print(f"ONNX 模型已导出: {onnx_path}")

        return {
            "status": "ok",
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history),
            "checkpoint_dir": str(self.checkpoint_dir),
        }
