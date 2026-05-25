from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.notifier import Notifier
from src.pipeline import PenaltyResearchPipeline
from src.time_utils import utc_now_iso


@dataclass(frozen=True)
class LiveFrame:
    frame_index: int
    video_ts: float | None
    frame_path: str
    captured_at: str


class VideoFrameSource:
    """实时视频源。

    支持摄像头编号、本地视频文件、HTTP/RTSP 直播流。真正的解码由 OpenCV 完成；
    如果没有安装 `opencv-python`，会在运行 live 命令时给出清晰错误。
    """

    def __init__(
        self,
        source: str,
        frames_dir: Path,
        sample_interval_sec: float = 1.0,
        save_frames: bool = True,
    ) -> None:
        self.source = source
        self.frames_dir = frames_dir
        self.sample_interval_sec = max(0.1, sample_interval_sec)
        self.save_frames = save_frames

    def frames(self, max_frames: int = 0) -> Iterator[LiveFrame]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("实时视频模式需要安装 opencv-python：pip install -r requirements.txt") from exc

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(_coerce_video_source(self.source))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频源：{self.source}")

        produced = 0
        try:
            while max_frames <= 0 or produced < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                produced += 1
                video_ts = _video_timestamp(capture, fallback_index=produced)
                frame_path = ""
                if self.save_frames:
                    frame_path = str(self.frames_dir / f"frame_{produced:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                yield LiveFrame(
                    frame_index=produced,
                    video_ts=video_ts,
                    frame_path=frame_path,
                    captured_at=utc_now_iso(),
                )
                time.sleep(self.sample_interval_sec)
        finally:
            capture.release()


class LiveStateProvider:
    """读取实时状态文件。

    状态文件是给外部视觉模型/人工标注/盘口模块写入的桥接接口。文件可以不存在；
    不存在时系统会用空信号继续跑，终端会持续给出低概率判断。
    """

    def __init__(
        self,
        path: Path | None,
        default_match_context: dict[str, Any] | None = None,
        default_market_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.default_match_context = default_match_context or {}
        self.default_market_snapshot = default_market_snapshot or {}

    def read(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.path is not None and self.path.exists():
            payload = _read_json_object(self.path)
        return {
            "signals": payload.get("signals", {}),
            "market_snapshot": payload.get("market_snapshot", self.default_market_snapshot),
            "match_context": payload.get("match_context", self.default_match_context),
        }


class LiveVideoRunner:
    """在线实时运行器：视频抽帧 -> 证据包 -> 预测 -> 纸面提醒。"""

    def __init__(
        self,
        pipeline: PenaltyResearchPipeline,
        frame_source: VideoFrameSource,
        state_provider: LiveStateProvider,
        match_id: str,
        source_label: str,
        print_all: bool = True,
        notifier: Notifier | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.frame_source = frame_source
        self.state_provider = state_provider
        self.match_id = match_id
        self.source_label = source_label
        self.print_all = print_all
        self.notifier = notifier or Notifier()

    def run(self, max_frames: int = 0) -> dict[str, Any]:
        total = 0
        actionable = 0
        paper_orders = 0
        for frame in self.frame_source.frames(max_frames=max_frames):
            total += 1
            state = self.state_provider.read()
            raw_event = {
                "event_id": f"live_{self.match_id}_{frame.frame_index:06d}",
                "match_id": self.match_id,
                "timestamp_utc": frame.captured_at,
                "video_ts": frame.video_ts,
                "source": self.source_label,
                "frame_paths": [frame.frame_path] if frame.frame_path else [],
                "signals": state["signals"],
                "market_snapshot": state["market_snapshot"],
                "match_context": state["match_context"],
            }
            result = self.pipeline.process_raw_event(raw_event)
            prediction = result["prediction"]
            is_actionable = (
                prediction["penalty_probability"] >= self.pipeline.settings.probability_threshold
                and prediction["confidence"] >= self.pipeline.settings.min_confidence
            )
            if is_actionable:
                actionable += 1
            if result["paper_order"] is not None:
                paper_orders += 1
            if self.print_all or is_actionable:
                print(format_live_line(frame.frame_index, result), flush=True)

            if is_actionable and result["paper_order"] is not None:
                prob = prediction["penalty_probability"]
                side = prediction["side"]
                self.notifier.maybe_notify(
                    title="Penalty Monitor - 疑似点球",
                    message=f"概率 {prob:.1%} | 方向 {side} | 比赛 {self.match_id}",
                    probability=prob,
                )

        summary = {
            "status": "ok",
            "mode": "live",
            "total_frames": total,
            "actionable_predictions": actionable,
            "paper_orders": paper_orders,
            "run_dir": str(self.pipeline.store.root),
        }
        self.pipeline.store.write_summary(summary)
        return summary


def format_live_line(frame_index: int, result: dict[str, Any]) -> str:
    prediction = result["prediction"]
    order_mark = "纸面订单=是" if result["paper_order"] is not None else "纸面订单=否"
    blocks = ",".join(result.get("execution_blocks") or [])
    if blocks:
        order_mark = f"{order_mark} 阻止原因={blocks}"
    reasons = ",".join(prediction.get("reason_codes") or [])
    return (
        f"[实时] frame={frame_index} "
        f"点球概率={prediction['penalty_probability']:.2%} "
        f"置信度={prediction['confidence']:.2%} "
        f"方向={prediction['side']} {order_mark} "
        f"原因={reasons or '-'}"
    )


def _coerce_video_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def _video_timestamp(capture: Any, fallback_index: int) -> float | None:
    try:
        import cv2

        value = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if value > 0:
            return round(value / 1000.0, 3)
    except Exception:
        pass
    return float(fallback_index)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}

