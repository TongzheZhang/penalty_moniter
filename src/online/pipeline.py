from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config import Settings
from src.live import LiveStateProvider, LiveVideoRunner, VideoFrameSource
from src.models import MatchContext
from src.notifier import Notifier
from src.online.feature_extractor import RealtimeFeatureExtractor
from src.online.inference_engine import PenaltyInferenceEngine
from src.online.signal_emitter import SignalEmitter
from src.pipeline import PenaltyResearchPipeline
from src.storage.jsonl_store import RunStore


class MLOnlinePipeline:
    """基于ML模型的在线点球监控流水线。

    替代原有规则引擎，使用训练好的神经网络进行实时推理。
    """

    def __init__(
        self,
        model_path: Path,
        settings: Settings | None = None,
        probability_threshold: float = 0.75,
        min_confidence: float = 0.55,
    ) -> None:
        self.settings = settings or Settings()
        self.inference_engine = PenaltyInferenceEngine(model_path=model_path)
        self.signal_emitter = SignalEmitter(
            probability_threshold=probability_threshold,
            min_confidence=min_confidence,
        )

    def watch_stream(
        self,
        stream_url: str,
        match_id: str,
        output_dir: Path | None = None,
        max_frames: int = 0,
        sample_interval_sec: float = 1.0,
        match_context: MatchContext | None = None,
    ) -> dict[str, Any]:
        """监控直播流，实时推理并输出信号。

        返回运行摘要。
        """
        store = RunStore(output_dir or self.settings.new_run_dir(Path(".")))
        frames_dir = store.root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_source = VideoFrameSource(
            source=stream_url,
            frames_dir=frames_dir,
            sample_interval_sec=sample_interval_sec,
            save_frames=True,
        )

        feature_extractor = RealtimeFeatureExtractor(
            frame_source=frame_source,
            match_context=match_context or MatchContext(),
        )

        total = 0
        triggered = 0
        max_prob = 0.0

        notifier = Notifier(self.settings.notify_config)
        self.signal_emitter.reset()

        for frame in frame_source.frames(max_frames=max_frames):
            total += 1
            packet = feature_extractor.tick(frame.frame_path)
            if packet is None:
                continue

            result = self.inference_engine.predict(
                visual=packet.visual,
                text=packet.text,
                context=packet.context,
            )

            signal = self.signal_emitter.emit(result, time.time())
            if signal.smoothed_probability > max_prob:
                max_prob = signal.smoothed_probability

            if signal.triggered:
                triggered += 1
                notifier.maybe_notify(
                    title="Penalty Monitor (ML) - 疑似点球",
                    message=f"概率 {signal.smoothed_probability:.1%} | 比赛 {match_id}",
                    probability=signal.smoothed_probability,
                )
                print(
                    f"[ML触发] frame={frame.frame_index} "
                    f"概率={signal.smoothed_probability:.2%} "
                    f"预计{signal.seconds_to_event or '?'}秒后发生 "
                    f"解说={packet.raw_commentary[:30]}..."
                )

        summary = {
            "status": "ok",
            "mode": "ml_online",
            "model": str(self.inference_engine.model_path),
            "match_id": match_id,
            "total_frames": total,
            "triggered_signals": triggered,
            "max_probability": round(max_prob, 4),
            "run_dir": str(store.root),
        }
        store.write_summary(summary)
        return summary
