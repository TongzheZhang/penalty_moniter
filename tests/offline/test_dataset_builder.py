import json
from pathlib import Path

from src.offline.dataset_builder import DatasetBuilder, DatasetSample


def test_dataset_sample_roundtrip() -> None:
    sample = DatasetSample(
        match_id="m1",
        slice_idx=0,
        start_sec=10.0,
        end_sec=20.0,
        slice_path="slices/slice_00000_10s_20s.mp4",
        label=1.0,
        seconds_to_penalty=5.0,
    )
    payload = sample.to_dict()
    assert payload["label"] == 1.0
    assert payload["seconds_to_penalty"] == 5.0


def test_dataset_builder_empty_matches(tmp_path: Path) -> None:
    builder = DatasetBuilder(matches_dir=tmp_path)
    stats = builder.build(output_dir=tmp_path / "output")
    assert stats["total"] == 0
