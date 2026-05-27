from pathlib import Path

from src.offline.preprocessor import MatchPreprocessor, TranscriptSegment


def test_transcript_segment_roundtrip() -> None:
    seg = TranscriptSegment(
        start_sec=10.5,
        end_sec=15.2,
        text="点球！",
        confidence=0.9,
    )
    payload = seg.to_dict()
    restored = TranscriptSegment.from_dict(payload)
    assert restored.text == "点球！"
    assert restored.start_sec == 10.5


def test_preprocessor_init(tmp_path: Path) -> None:
    # 没有视频文件时应返回None
    preprocessor = MatchPreprocessor(match_dir=tmp_path)
    assert preprocessor.video_path is None
