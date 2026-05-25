from pathlib import Path

from src.live import LiveStateProvider, format_live_line


def test_live_state_provider_uses_defaults_when_file_missing(tmp_path: Path) -> None:
    provider = LiveStateProvider(
        path=tmp_path / "missing.json",
        default_match_context={"home": "A", "away": "B"},
        default_market_snapshot={"market_id": "m1"},
    )

    state = provider.read()

    assert state["signals"] == {}
    assert state["match_context"]["home"] == "A"
    assert state["market_snapshot"]["market_id"] == "m1"


def test_format_live_line_is_chinese_friendly() -> None:
    line = format_live_line(
        3,
        {
            "prediction": {
                "penalty_probability": 0.81,
                "confidence": 0.77,
                "side": "home",
                "reason_codes": ["ref_var_walk_high"],
            },
            "paper_order": {"event_id": "evt"},
            "execution_blocks": [],
        },
    )

    assert "点球概率=81.00%" in line
    assert "纸面订单=是" in line
