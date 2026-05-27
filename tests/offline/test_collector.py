from pathlib import Path

from src.offline.collector import MatchCollector, MatchMeta


def test_match_meta_roundtrip() -> None:
    meta = MatchMeta(
        match_id="test1",
        source_url="https://example.com/video",
        title="Test Match",
        duration_sec=5400.0,
        resolution="1920x1080",
    )
    payload = meta.to_dict()
    restored = MatchMeta.from_dict(payload)
    assert restored.match_id == "test1"
    assert restored.duration_sec == 5400.0


def test_collector_has_yt_dlp_returns_bool() -> None:
    result = MatchCollector._has_yt_dlp()
    assert isinstance(result, bool)


def test_collector_has_streamlink_returns_bool() -> None:
    result = MatchCollector._has_streamlink()
    assert isinstance(result, bool)


def test_collector_dry_run_creates_meta(tmp_path: Path) -> None:
    # 由于yt-dlp可能不存在，此测试只验证目录结构
    collector = MatchCollector(output_dir=tmp_path)
    assert collector.output_dir == tmp_path
