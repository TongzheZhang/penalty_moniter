from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MatchMeta:
    match_id: str
    source_url: str
    title: str = ""
    duration_sec: float | None = None
    resolution: str = ""
    downloaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_size_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "source_url": self.source_url,
            "title": self.title,
            "duration_sec": self.duration_sec,
            "resolution": self.resolution,
            "downloaded_at": self.downloaded_at,
            "file_size_mb": self.file_size_mb,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MatchMeta":
        return cls(
            match_id=str(payload.get("match_id", "")),
            source_url=str(payload.get("source_url", "")),
            title=str(payload.get("title", "")),
            duration_sec=payload.get("duration_sec"),
            resolution=str(payload.get("resolution", "")),
            downloaded_at=str(payload.get("downloaded_at", "")),
            file_size_mb=payload.get("file_size_mb"),
        )


class MatchCollector:
    """比赛录像采集器。

    支持 yt-dlp（优先）和 streamlink（fallback）下载公开可访问的视频。
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path("data/matches")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _has_yt_dlp() -> bool:
        try:
            subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    @staticmethod
    def _has_streamlink() -> bool:
        try:
            subprocess.run(["streamlink", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _parse_info_yt_dlp(self, url: str) -> dict[str, Any] | None:
        try:
            result = subprocess.run(
                ["yt-dlp", "--dump-json", "--skip-download", url],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            return json.loads(result.stdout.splitlines()[0])
        except Exception:
            return None

    def collect_info(self, url: str) -> MatchMeta | None:
        """只采集元数据，不下载视频（dry-run 模式）。"""
        info = self._parse_info_yt_dlp(url)
        if not info:
            return None
        return MatchMeta(
            match_id=info.get("id", "unknown"),
            source_url=url,
            title=info.get("title", ""),
            duration_sec=info.get("duration"),
            resolution=info.get("resolution", ""),
        )

    def download_match(
        self,
        url: str,
        match_id: str | None = None,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> Path | None:
        """下载单场比赛录像。

        返回下载目录路径；dry_run 模式下只解析元数据并创建目录。
        """
        info = self._parse_info_yt_dlp(url)
        if not info:
            raise RuntimeError(f"无法解析视频信息: {url}")

        resolved_id = match_id or info.get("id", "unknown")
        target_dir = (output_dir or self.output_dir) / resolved_id
        target_dir.mkdir(parents=True, exist_ok=True)

        meta = MatchMeta(
            match_id=resolved_id,
            source_url=url,
            title=info.get("title", ""),
            duration_sec=info.get("duration"),
            resolution=info.get("resolution", ""),
        )

        meta_path = target_dir / "meta.json"
        meta_path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        if dry_run:
            return target_dir

        output_template = str(target_dir / "match")
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "-f", "best[height<=720]/best",  # 限制720p避免过大
                    "--merge-output-format", "mp4",
                    "-o", output_template + ".%(ext)s",
                    url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"yt-dlp 下载失败: {exc.stderr.decode()}") from exc

        # 更新文件大小
        video_files = list(target_dir.glob("match.*"))
        if video_files:
            meta = MatchMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
            meta = MatchMeta(
                match_id=meta.match_id,
                source_url=meta.source_url,
                title=meta.title,
                duration_sec=meta.duration_sec,
                resolution=meta.resolution,
                downloaded_at=meta.downloaded_at,
                file_size_mb=round(video_files[0].stat().st_size / (1024 * 1024), 2),
            )
            meta_path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        return target_dir

    def batch_download(
        self,
        urls: list[str],
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> list[Path]:
        """批量下载。"""
        results: list[Path] = []
        for i, url in enumerate(urls):
            try:
                path = self.download_match(url, match_id=f"batch_{i:03d}", output_dir=output_dir, dry_run=dry_run)
                if path:
                    results.append(path)
            except RuntimeError as exc:
                print(f"跳过下载失败项 ({url}): {exc}")
        return results
