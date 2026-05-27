from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.commentary import CommentaryMonitor
from src.live import VideoFrameSource
from src.models import MatchContext


@dataclass
class FeaturePacket:
    """组装好的特征包，直接送入推理引擎。"""
    visual: np.ndarray    # (1, T, D_v)
    text: np.ndarray      # (1, T, D_t)
    context: np.ndarray   # (1, D_c)
    timestamp: float
    raw_commentary: str = ""


class RealtimeFeatureExtractor:
    """实时特征提取器。

    从直播流中持续提取视觉、文本、上下文特征，组装为推理所需的 FeaturePacket。
    """

    def __init__(
        self,
        frame_source: VideoFrameSource,
        commentary_monitor: CommentaryMonitor | None = None,
        match_context: MatchContext | None = None,
        seq_len: int = 6,
        visual_dim: int = 512,
        text_dim: int = 768,
    ) -> None:
        self.frame_source = frame_source
        self.commentary_monitor = commentary_monitor
        self.match_context = match_context or MatchContext()
        self.seq_len = seq_len
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        # 滑动窗口缓冲区
        self._visual_buffer: deque[np.ndarray] = deque(maxlen=seq_len)
        self._text_buffer: deque[np.ndarray] = deque(maxlen=seq_len)
        self._last_extraction = 0.0

    def _extract_visual_feature(self, frame_path: str) -> np.ndarray:
        """从单帧提取视觉特征（占位实现，实际应加载预训练ResNet）。"""
        # MVP阶段：用随机初始化占位，后续替换为真实特征提取
        return np.random.randn(self.visual_dim).astype(np.float32) * 0.1

    def _extract_text_feature(self, text: str) -> np.ndarray:
        """从文本提取特征（占位实现，实际应加载预训练BERT）。"""
        # MVP阶段：用随机初始化占位
        return np.random.randn(self.text_dim).astype(np.float32) * 0.1

    def _build_context_vector(self) -> np.ndarray:
        """构建上下文特征向量。"""
        ctx = np.zeros(16, dtype=np.float32)
        ctx[0] = (self.match_context.minute or 0) / 120.0
        ctx[1] = (self.match_context.score_home or 0) / 10.0
        ctx[2] = (self.match_context.score_away or 0) / 10.0
        ctx[3] = self.match_context.red_cards_home / 5.0
        ctx[4] = self.match_context.red_cards_away / 5.0
        return ctx.reshape(1, -1)

    def tick(self, frame_path: str = "") -> FeaturePacket | None:
        """每帧调用，提取当前时刻特征并组装 Packet。

        如果缓冲区未满，返回 None。
        """
        now = time.time()
        if now - self._last_extraction < 1.0:
            return None
        self._last_extraction = now

        # 视觉特征
        vis = self._extract_visual_feature(frame_path)
        self._visual_buffer.append(vis)

        # 文本特征
        text_feature = np.zeros(self.text_dim, dtype=np.float32)
        raw_commentary = ""
        if self.commentary_monitor is not None:
            recent = self.commentary_monitor.get_recent_transcripts(max_age_sec=5.0)
            if recent:
                combined_text = " ".join([t.text for t in recent])
                raw_commentary = combined_text
                text_feature = self._extract_text_feature(combined_text)
        self._text_buffer.append(text_feature)

        if len(self._visual_buffer) < self.seq_len:
            return None

        visual_seq = np.stack(list(self._visual_buffer), axis=0).reshape(1, self.seq_len, self.visual_dim)
        text_seq = np.stack(list(self._text_buffer), axis=0).reshape(1, self.seq_len, self.text_dim)
        context = self._build_context_vector()

        return FeaturePacket(
            visual=visual_seq,
            text=text_seq,
            context=context,
            timestamp=now,
            raw_commentary=raw_commentary,
        )
