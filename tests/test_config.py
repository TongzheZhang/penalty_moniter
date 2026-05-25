from src.config import Settings


def test_settings_defaults() -> None:
    s = Settings()
    assert s.mode == "paper"
    assert s.probability_threshold == 0.75
    assert s.cooldown_sec == 0.0
    assert s.enable_real_trading is False


def test_settings_assert_paper_only() -> None:
    s = Settings()
    s.assert_paper_only()  # 不应抛出异常


def test_settings_assert_paper_only_fails_on_real_trading() -> None:
    s = Settings(enable_real_trading=True)
    try:
        s.assert_paper_only()
        assert False, "应该抛出 ValueError"
    except ValueError as exc:
        assert "真钱交易" in str(exc)


def test_settings_from_dict() -> None:
    payload = {
        "project_name": "test_proj",
        "decision": {"probability_threshold": 0.8, "min_confidence": 0.6},
        "paper": {"cooldown_sec": 30.0},
    }
    s = Settings.from_dict(payload)
    assert s.project_name == "test_proj"
    assert s.probability_threshold == 0.8
    assert s.min_confidence == 0.6
    assert s.cooldown_sec == 30.0
