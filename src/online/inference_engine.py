from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class InferenceResult:
    probability: float
    confidence: float
    seconds_to_event: float | None
    model_version: str = "unknown"


class PenaltyInferenceEngine:
    """点球预测推理引擎。

    支持 ONNX Runtime（优先）和 PyTorch 两种后端。
    """

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._backend: str | None = None
        self._session: Any = None
        self._torch_model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.suffix == ".onnx":
            self._load_onnx()
        elif self.model_path.suffix == ".pt":
            self._load_torch()
        else:
            raise ValueError(f"不支持的模型格式: {self.model_path.suffix}")

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime 未安装。请运行: pip install onnxruntime"
            ) from exc

        providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._backend = "onnx"

    def _load_torch(self) -> None:
        import torch
        from src.offline.model import ModelConfig, PenaltyPredictor

        predictor = PenaltyPredictor(config=ModelConfig())
        self._torch_model = predictor._build_module().to(self.device)
        self._torch_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self._torch_model.eval()
        self._backend = "torch"

    def predict(
        self,
        visual: np.ndarray,
        text: np.ndarray,
        context: np.ndarray,
    ) -> InferenceResult:
        """推理单条样本。

        Args:
            visual: (1, T, D_v) 视觉特征数组
            text: (1, T, D_t) 文本特征数组
            context: (1, D_c) 上下文特征数组
        """
        if self._backend == "onnx":
            return self._predict_onnx(visual, text, context)
        return self._predict_torch(visual, text, context)

    def _predict_onnx(
        self,
        visual: np.ndarray,
        text: np.ndarray,
        context: np.ndarray,
    ) -> InferenceResult:
        inputs = {
            "visual": visual.astype(np.float32),
            "text": text.astype(np.float32),
            "context": context.astype(np.float32),
        }
        prob, time_to = self._session.run(None, inputs)
        prob_val = float(1 / (1 + np.exp(-prob[0])))  # sigmoid
        time_val = float(time_to[0]) if time_to is not None else None

        return InferenceResult(
            probability=round(prob_val, 4),
            confidence=round(abs(prob_val - 0.5) * 2, 4),  # 距离0.5越远越确信
            seconds_to_event=round(time_val, 2) if time_val is not None else None,
        )

    def _predict_torch(
        self,
        visual: np.ndarray,
        text: np.ndarray,
        context: np.ndarray,
    ) -> InferenceResult:
        import torch

        v = torch.from_numpy(visual).float().to(self.device)
        t = torch.from_numpy(text).float().to(self.device)
        c = torch.from_numpy(context).float().to(self.device)

        with torch.no_grad():
            prob, time_to = self._torch_model(v, t, c)
            prob_val = float(torch.sigmoid(prob).cpu().item())
            time_val = float(time_to.cpu().item())

        return InferenceResult(
            probability=round(prob_val, 4),
            confidence=round(abs(prob_val - 0.5) * 2, 4),
            seconds_to_event=round(time_val, 2),
        )
