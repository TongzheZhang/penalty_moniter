from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models import CommentaryTranscript
from src.time_utils import utc_now_iso


@dataclass(frozen=True)
class CommentaryResult:
    urgency_score: float
    keywords_hit: list[str]
    matched: bool


class CommentaryAnalyzer:
    """纯规则引擎：根据关键词词典分析解说文本的点球紧急程度。

    零外部依赖，可在任何环境中独立运行。
    """

    DEFAULT_KEYWORDS: dict[str, list[str]] = {
        "strong": [
            "点球", "penalty", "十二码", "VAR 介入", "VAR介入",
            "犯规在禁区", "禁区犯规", "判点球", "给了点球", "吹点球",
        ],
        "medium": [
            "禁区里", "禁区内", "倒地", "裁判跑向", "走向屏幕",
            "VAR", "视频助理", "慢动作", "回放", "看回放",
            "手球", "拉人", "推人",
        ],
        "weak": [
            "有争议", "身体接触", "战术犯规", "拉扯", "对抗",
            "侵犯", "疑似", "可能", "看起来",
        ],
    }

    DEFAULT_WEIGHTS: dict[str, float] = {"strong": 1.0, "medium": 0.6, "weak": 0.3}

    def __init__(
        self,
        keywords: dict[str, list[str]] | None = None,
        weights: dict[str, float] | None = None,
        min_urgency: float = 0.5,
    ) -> None:
        self.keywords = keywords or dict(self.DEFAULT_KEYWORDS)
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self.min_urgency = min_urgency
        # 预编译正则，提高匹配速度
        self._patterns: dict[str, list[re.Pattern]] = {
            tier: [re.compile(re.escape(kw)) for kw in kws]
            for tier, kws in self.keywords.items()
        }

    def analyze(self, text: str) -> CommentaryResult:
        if not text:
            return CommentaryResult(urgency_score=0.0, keywords_hit=[], matched=False)

        hit_keywords: list[str] = []
        max_score = 0.0
        for tier, patterns in self._patterns.items():
            weight = self.weights.get(tier, 0.0)
            for pat in patterns:
                if pat.search(text):
                    kw = pat.pattern.replace("\\", "")  # unescape
                    hit_keywords.append(kw)
                    max_score = max(max_score, weight)

        urgency = max_score
        matched = urgency >= self.min_urgency
        return CommentaryResult(
            urgency_score=round(urgency, 4),
            keywords_hit=hit_keywords,
            matched=matched,
        )

    def analyze_batch(self, texts: list[str]) -> list[CommentaryResult]:
        return [self.analyze(t) for t in texts]


class AudioStreamExtractor:
    """使用 ffmpeg 从视频流中提取音频片段。"""

    def __init__(self, stream_url: str, output_dir: Path) -> None:
        self.stream_url = stream_url
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_ok: bool | None = None

    def _check_ffmpeg(self) -> bool:
        if self._ffmpeg_ok is not None:
            return self._ffmpeg_ok
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            self._ffmpeg_ok = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._ffmpeg_ok = False
        return self._ffmpeg_ok

    def extract_segment(self, duration_sec: float = 3.0) -> Path | None:
        """提取最近 ``duration_sec`` 秒的音频到 WAV 文件。

        返回临时 WAV 文件路径；如果 ffmpeg 不可用则返回 None。
        """
        if not self._check_ffmpeg():
            return None

        out_path = self.output_dir / f"audio_{int(time.time() * 1000)}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", self.stream_url,
            "-t", str(duration_sec),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(out_path),
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=duration_sec + 5.0,
                check=True,
            )
            return out_path if out_path.exists() else None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None


class WhisperTranscriber:
    """faster-whisper 封装，延迟加载模型。"""

    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper 未安装，无法使用 audio 解说模式。"
                "请运行: pip install faster-whisper，或改用 --commentary-mode file"
            ) from exc
        self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: Path) -> list[dict[str, Any]]:
        """返回转录片段列表，每个片段包含 text, start, end, confidence。"""
        model = self._ensure_model()
        segments, _info = model.transcribe(str(audio_path), beam_size=5, language="zh")
        results = []
        for seg in segments:
            results.append({
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
                "confidence": getattr(seg, "avg_logprob", 0.0),
            })
        return results


class CommentaryMonitor:
    """解说监控协调器，支持 file / audio 双模式，独立线程运行。

    - **file 模式**：定期读取外部文本文件的新增行，逐行分析。
    - **audio 模式**：定期从视频流提取音频片段，STT 转录后分析。
    """

    def __init__(
        self,
        mode: str = "off",
        stream_url: str | None = None,
        commentary_file: Path | None = None,
        analyzer: CommentaryAnalyzer | None = None,
        interval_sec: float = 3.0,
        whisper_model: str = "base",
        work_dir: Path | None = None,
    ) -> None:
        self.mode = mode
        self.interval_sec = max(0.5, interval_sec)
        self.analyzer = analyzer or CommentaryAnalyzer()
        self.work_dir = work_dir or Path("data/commentary_work")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._latest_score = 0.0
        self._transcripts: list[CommentaryTranscript] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # file mode state
        self._file_path: Path | None = commentary_file
        self._file_offset = 0

        # audio mode state
        self._extractor: AudioStreamExtractor | None = None
        self._transcriber: WhisperTranscriber | None = None
        if mode == "audio":
            if not stream_url:
                raise ValueError("audio 模式必须提供 stream_url")
            self._extractor = AudioStreamExtractor(stream_url, self.work_dir / "audio")
            self._transcriber = WhisperTranscriber(model_size=whisper_model)
        elif mode == "file":
            if not commentary_file:
                raise ValueError("file 模式必须提供 commentary_file")

    def start(self) -> None:
        if self.mode == "off":
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.mode == "audio":
                    self._tick_audio()
                elif self.mode == "file":
                    self._tick_file()
            except Exception:
                # 单周期错误不应终止监控线程
                pass
            self._stop_event.wait(self.interval_sec)

    def _tick_file(self) -> None:
        if self._file_path is None or not self._file_path.exists():
            return
        text = self._file_path.read_text(encoding="utf-8")
        if len(text) < self._file_offset:
            # 文件被截断，从头开始
            self._file_offset = 0
        new_text = text[self._file_offset:]
        self._file_offset = len(text)

        for line in new_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            result = self.analyzer.analyze(line)
            transcript = CommentaryTranscript(
                timestamp_utc=utc_now_iso(),
                text=line,
                keywords_hit=result.keywords_hit,
                urgency_score=result.urgency_score,
            )
            with self._lock:
                self._transcripts.append(transcript)
                if result.matched:
                    self._latest_score = max(self._latest_score, result.urgency_score)
                # 滑动窗口衰减：保留最近 60 条
                if len(self._transcripts) > 60:
                    self._transcripts = self._transcripts[-60:]

    def _tick_audio(self) -> None:
        if self._extractor is None or self._transcriber is None:
            return
        audio_path = self._extractor.extract_segment(duration_sec=self.interval_sec)
        if audio_path is None:
            return
        try:
            segments = self._transcriber.transcribe(audio_path)
        finally:
            # 清理临时音频文件
            try:
                audio_path.unlink()
            except OSError:
                pass

        for seg in segments:
            text = seg.get("text", "")
            if not text:
                continue
            result = self.analyzer.analyze(text)
            transcript = CommentaryTranscript(
                timestamp_utc=utc_now_iso(),
                video_ts=seg.get("start"),
                text=text,
                keywords_hit=result.keywords_hit,
                urgency_score=result.urgency_score,
                raw_confidence=seg.get("confidence", 0.0),
            )
            with self._lock:
                self._transcripts.append(transcript)
                if result.matched:
                    self._latest_score = max(self._latest_score, result.urgency_score)
                if len(self._transcripts) > 60:
                    self._transcripts = self._transcripts[-60:]

    def get_latest_score(self) -> float:
        """获取当前解说紧急度分数 [0,1]。"""
        with self._lock:
            return self._latest_score

    def get_recent_transcripts(self, max_age_sec: float = 30.0) -> list[CommentaryTranscript]:
        """获取最近 ``max_age_sec`` 秒内的转录记录。"""
        with self._lock:
            # file 模式没有 video_ts，用列表长度近似（简化）
            # 更精确的做法：在 CommentaryTranscript 中记录入库时间戳
            cutoff_count = int(max_age_sec / self.interval_sec) + 1
            return list(self._transcripts[-cutoff_count:])

    def reset_score(self) -> None:
        """手动重置最新分数（例如触发纸面订单后）。"""
        with self._lock:
            self._latest_score = 0.0

    def all_transcripts(self) -> list[CommentaryTranscript]:
        with self._lock:
            return list(self._transcripts)
