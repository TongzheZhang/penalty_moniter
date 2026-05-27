from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.agents.commentary import WhisperTranscriber
from src.time_utils import utc_now_iso


@dataclass
class VideoSlice:
    idx: int
    start_sec: float
    end_sec: float
    path: Path


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            start_sec=float(payload.get("start_sec", 0)),
            end_sec=float(payload.get("end_sec", 0)),
            text=str(payload.get("text", "")),
            confidence=float(payload.get("confidence", 0)),
        )


class MatchPreprocessor:
    """比赛录像预处理管道。

    负责：视频切片 → 关键帧提取 → 音频提取 → STT转录。
    """

    def __init__(self, match_dir: Path, whisper_model: str = "base") -> None:
        self.match_dir = Path(match_dir)
        self.video_path = self._find_video_file()
        self.slices_dir = self.match_dir / "slices"
        self.features_dir = self.match_dir / "features"
        self.whisper_model = whisper_model

    def _find_video_file(self) -> Path | None:
        for ext in (".mp4", ".mkv", ".flv", ".ts"):
            candidates = list(self.match_dir.glob(f"match{ext}"))
            if candidates:
                return candidates[0]
        return None

    def _ffmpeg_available(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def slice_video(
        self,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
    ) -> Iterator[VideoSlice]:
        """将视频切分为重叠片段。"""
        if self.video_path is None or not self.video_path.exists():
            raise FileNotFoundError(f"未找到视频文件: {self.match_dir}")
        if not self._ffmpeg_available():
            raise RuntimeError("ffmpeg 未安装")

        self.slices_dir.mkdir(parents=True, exist_ok=True)

        # 获取视频时长
        duration = self._get_video_duration()
        if duration is None or duration <= 0:
            raise RuntimeError("无法获取视频时长")

        idx = 0
        start = 0.0
        while start + window_sec <= duration:
            end = start + window_sec
            out_path = self.slices_dir / f"slice_{idx:05d}_{start:.0f}s_{end:.0f}s.mp4"
            if not out_path.exists():
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-t", str(window_sec),
                    "-i", str(self.video_path),
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    str(out_path),
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
                except subprocess.CalledProcessError:
                    pass  # 单片段失败不终止整体流程

            if out_path.exists():
                yield VideoSlice(idx=idx, start_sec=start, end_sec=end, path=out_path)

            idx += 1
            start += stride_sec

    def _get_video_duration(self) -> float | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    str(self.video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except Exception:
            return None

    def extract_keyframes(self, slice_path: Path, n_frames: int = 4) -> list[Path]:
        """从单个切片提取均匀分布的关键帧。"""
        if not self._ffmpeg_available():
            raise RuntimeError("ffmpeg 未安装")

        out_dir = self.features_dir / "keyframes" / slice_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 获取切片时长
        duration = self._get_media_duration(slice_path)
        if duration is None or duration <= 0:
            duration = 10.0

        frames: list[Path] = []
        for i in range(n_frames):
            ts = duration * i / max(1, n_frames - 1)
            out_path = out_dir / f"frame_{i:02d}_{ts:.2f}s.jpg"
            if not out_path.exists():
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(ts),
                    "-i", str(slice_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(out_path),
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=10)
                except subprocess.CalledProcessError:
                    continue
            if out_path.exists():
                frames.append(out_path)

        return frames

    def _get_media_duration(self, path: Path) -> float | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except Exception:
            return None

    def extract_audio(self) -> Path:
        """提取完整视频音轨为 WAV。"""
        if self.video_path is None or not self.video_path.exists():
            raise FileNotFoundError(f"未找到视频文件: {self.match_dir}")
        if not self._ffmpeg_available():
            raise RuntimeError("ffmpeg 未安装")

        out_path = self.match_dir / "audio.wav"
        if out_path.exists():
            return out_path

        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(out_path),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=300)
        return out_path

    def transcribe_audio(self, audio_path: Path | None = None) -> Path:
        """对音轨进行 STT 转录，输出 transcript.jsonl。"""
        audio = audio_path or self.extract_audio()
        out_path = self.match_dir / "transcript.jsonl"
        if out_path.exists():
            return out_path

        transcriber = WhisperTranscriber(model_size=self.whisper_model)
        segments = transcriber.transcribe(audio)

        with out_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                ts = TranscriptSegment(
                    start_sec=seg.get("start", 0.0),
                    end_sec=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    confidence=seg.get("confidence", 0.0),
                )
                f.write(json.dumps(ts.to_dict(), ensure_ascii=False) + "\n")

        return out_path

    def process_all(
        self,
        slice_window: float = 10.0,
        slice_stride: float = 5.0,
        keyframes_per_slice: int = 4,
    ) -> dict[str, Any]:
        """一键预处理：切片 + 关键帧 + 音频 + STT。"""
        summary: dict[str, Any] = {
            "match_dir": str(self.match_dir),
            "video_path": str(self.video_path) if self.video_path else None,
            "slices": 0,
            "keyframes": 0,
            "transcript_segments": 0,
            "started_at": utc_now_iso(),
        }

        # 1. 切片
        slices = list(self.slice_video(window_sec=slice_window, stride_sec=slice_stride))
        summary["slices"] = len(slices)

        # 2. 关键帧
        total_frames = 0
        for sl in slices:
            frames = self.extract_keyframes(sl.path, n_frames=keyframes_per_slice)
            total_frames += len(frames)
        summary["keyframes"] = total_frames

        # 3. 音频 + STT
        transcript_path = self.transcribe_audio()
        segs = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary["transcript_segments"] = len(segs)

        summary["finished_at"] = utc_now_iso()

        # 保存摘要
        summary_path = self.match_dir / "preprocess_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return summary
