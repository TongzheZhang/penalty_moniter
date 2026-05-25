from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from typing import Any


class StreamResolver(ABC):
    """直播流地址解析器抽象接口。

    将网页 URL（如 YouTube、B站、抖音）解析为可直接播放的流地址。
    不直接处理反爬逻辑，而是委托给外部工具（yt-dlp、streamlink 等）。
    """

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """判断此解析器是否能处理给定的 URL。"""
        ...

    @abstractmethod
    def resolve(self, url: str) -> str:
        """将网页 URL 解析为直接流地址。"""
        ...


class DirectUrlResolver(StreamResolver):
    """已经是直接流地址（RTMP/RTSP/HTTP-FLV/HLS），无需解析。"""

    DIRECT_SCHEMES = ("rtmp://", "rtsp://", "http://", "https://")

    def can_handle(self, url: str) -> bool:
        # 简单启发式：如果 URL 以常见流协议开头且没有明显网页特征，认为是直接地址
        if not url.startswith(self.DIRECT_SCHEMES):
            return False
        # 如果路径以 .flv/.m3u8/.ts 结尾，更可能是直接流
        lower = url.lower()
        if any(lower.endswith(ext) for ext in (".flv", ".m3u8", ".ts", ".mp4")):
            return True
        # 如果有查询参数包含 stream/play/live，也认为是直接流
        if any(kw in lower for kw in ("stream", "play", "live", "txSecret", "txTime")):
            return True
        return False

    def resolve(self, url: str) -> str:
        return url


class YtDlpResolver(StreamResolver):
    """通过 yt-dlp 获取最佳格式的流地址。"""

    def can_handle(self, url: str) -> bool:
        # yt-dlp 几乎支持所有主流平台
        return url.startswith(("http://", "https://"))

    def resolve(self, url: str) -> str:
        try:
            result = subprocess.run(
                ["yt-dlp", "-g", "-f", "best[height<=720]", url],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError("yt-dlp 未返回流地址")
            return lines[0]
        except FileNotFoundError as exc:
            raise RuntimeError(
                "yt-dlp 未安装。请先安装: pip install yt-dlp"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"yt-dlp 解析失败: {exc.stderr}") from exc


class StreamlinkResolver(StreamResolver):
    """通过 streamlink 获取流地址。"""

    def can_handle(self, url: str) -> bool:
        return url.startswith(("http://", "https://"))

    def resolve(self, url: str) -> str:
        try:
            result = subprocess.run(
                ["streamlink", "--stream-url", url, "best"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            addr = result.stdout.strip()
            if not addr:
                raise RuntimeError("streamlink 未返回流地址")
            return addr
        except FileNotFoundError as exc:
            raise RuntimeError(
                "streamlink 未安装。请先安装: pip install streamlink"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"streamlink 解析失败: {exc.stderr}") from exc


class FallbackResolver(StreamResolver):
    """兜底解析器：尝试多个子解析器，返回第一个成功的结果。"""

    def __init__(self, resolvers: list[StreamResolver] | None = None) -> None:
        self.resolvers = resolvers or [
            DirectUrlResolver(),
            StreamlinkResolver(),
            YtDlpResolver(),
        ]

    def can_handle(self, url: str) -> bool:
        return any(r.can_handle(url) for r in self.resolvers)

    def resolve(self, url: str) -> str:
        for resolver in self.resolvers:
            if resolver.can_handle(url):
                try:
                    return resolver.resolve(url)
                except RuntimeError:
                    continue
        raise RuntimeError(
            f"无法解析流地址: {url}\n"
            "提示：如果是国内平台，建议先用外部工具获取直接流地址，"
            "再以 RTMP/HTTP-FLV 形式传入 --video-source。"
        )


def resolve_stream_url(url: str) -> str:
    """便捷函数：使用默认 FallbackResolver 解析流地址。"""
    return FallbackResolver().resolve(url)
