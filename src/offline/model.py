from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class ModelConfig:
    visual_dim: int = 512
    text_dim: int = 768
    context_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    fusion_type: str = "lstm"  # lstm | transformer
    pos_weight: float = 3.0
    aux_loss_weight: float = 0.2


class PenaltyPredictor:
    """点球预测多模态模型。

    输入：视觉特征 (B, T, D_v) + 文本特征 (B, T, D_t) + 上下文 (B, D_c)
    输出：点球概率 [0,1] + 距点球秒数（回归）
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch 未安装。请运行: pip install torch torchvision"
            )
        self.config = config or ModelConfig()
        self._build_model()

    def _build_model(self) -> None:
        cfg = self.config

        # 特征投影层
        self.visual_proj = nn.Linear(cfg.visual_dim, cfg.hidden_dim)
        self.text_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.hidden_dim)

        # 时序融合
        if cfg.fusion_type == "lstm":
            self.fusion = nn.LSTM(
                input_size=cfg.hidden_dim * 2,
                hidden_size=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0,
                bidirectional=True,
            )
            fusion_out = cfg.hidden_dim * 2
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim * 2,
                nhead=4,
                dim_feedforward=cfg.hidden_dim * 4,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
            fusion_out = cfg.hidden_dim * 2

        # 输出头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_out + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(fusion_out + cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        self._model = None

    def forward(
        self,
        visual: Any,   # (B, T, D_v)
        text: Any,     # (B, T, D_t)
        context: Any,  # (B, D_c)
        mask: Any | None = None,
    ) -> tuple[Any, Any]:
        """返回 (probability, seconds_to_penalty)。"""
        # 投影
        v = self.visual_proj(visual)   # (B, T, H)
        t = self.text_proj(text)       # (B, T, H)
        c = self.context_proj(context) # (B, H)

        # 拼接视觉+文本
        features = torch.cat([v, t], dim=-1)  # (B, T, H*2)

        # 时序融合
        if self.config.fusion_type == "lstm":
            fused, _ = self.fusion(features)  # (B, T, H*2)
        else:
            if mask is not None:
                fused = self.fusion(features, src_key_padding_mask=~mask.bool())
            else:
                fused = self.fusion(features)

        # 取最后一个时间步 + 上下文
        last = fused[:, -1, :]  # (B, H*2)
        combined = torch.cat([last, c], dim=-1)  # (B, H*2 + H)

        prob = torch.sigmoid(self.classifier(combined)).squeeze(-1)  # (B,)
        time_to_event = self.regressor(combined).squeeze(-1)         # (B,)

        return prob, time_to_event

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return self.forward(*args, **kwargs)

    def save(self, path: Path) -> None:
        if self._model is None:
            self._model = self._build_module()
        torch.save(self._model.state_dict(), path)

    def load(self, path: Path) -> None:
        if self._model is None:
            self._model = self._build_module()
        self._model.load_state_dict(torch.load(path, map_location="cpu"))

    def _build_module(self) -> Any:
        """将模型组件组装为 nn.Module。"""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch 未安装")

        class _PenaltyPredictorModule(nn.Module):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__()
                self.visual_proj = nn.Linear(config.visual_dim, config.hidden_dim)
                self.text_proj = nn.Linear(config.text_dim, config.hidden_dim)
                self.context_proj = nn.Linear(config.context_dim, config.hidden_dim)

                if config.fusion_type == "lstm":
                    self.fusion = nn.LSTM(
                        input_size=config.hidden_dim * 2,
                        hidden_size=config.hidden_dim,
                        num_layers=config.num_layers,
                        batch_first=True,
                        dropout=config.dropout if config.num_layers > 1 else 0,
                        bidirectional=True,
                    )
                    fusion_out = config.hidden_dim * 2
                else:
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=config.hidden_dim * 2,
                        nhead=4,
                        dim_feedforward=config.hidden_dim * 4,
                        dropout=config.dropout,
                        batch_first=True,
                    )
                    self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
                    fusion_out = config.hidden_dim * 2

                self.classifier = nn.Sequential(
                    nn.Linear(fusion_out + config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
                )
                self.regressor = nn.Sequential(
                    nn.Linear(fusion_out + config.hidden_dim, config.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim // 2, 1),
                )
                self.config = config

            def forward(self, visual, text, context, mask=None):
                v = self.visual_proj(visual)
                t = self.text_proj(text)
                c = self.context_proj(context)
                features = torch.cat([v, t], dim=-1)
                if self.config.fusion_type == "lstm":
                    fused, _ = self.fusion(features)
                else:
                    if mask is not None:
                        fused = self.fusion(features, src_key_padding_mask=~mask.bool())
                    else:
                        fused = self.fusion(features)
                last = fused[:, -1, :]
                combined = torch.cat([last, c], dim=-1)
                prob = torch.sigmoid(self.classifier(combined)).squeeze(-1)
                time_to_event = self.regressor(combined).squeeze(-1)
                return prob, time_to_event

        module = _PenaltyPredictorModule(self.config)
        return module

    def to_onnx(self, path: Path, dummy_visual: Any, dummy_text: Any, dummy_context: Any) -> None:
        """导出 ONNX 模型。"""
        module = self._build_module()
        module.eval()
        torch.onnx.export(
            module,
            (dummy_visual, dummy_text, dummy_context),
            str(path),
            input_names=["visual", "text", "context"],
            output_names=["probability", "seconds_to_event"],
            dynamic_axes={
                "visual": {0: "batch", 1: "time"},
                "text": {0: "batch", 1: "time"},
                "context": {0: "batch"},
                "probability": {0: "batch"},
                "seconds_to_event": {0: "batch"},
            },
            opset_version=14,
        )
