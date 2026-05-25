from src.notifier import NotifyConfig, Notifier


def test_notify_config_defaults() -> None:
    cfg = NotifyConfig()
    assert cfg.enabled is True
    assert cfg.desktop is True
    assert cfg.sound is True
    assert cfg.min_probability == 0.75


def test_notify_config_from_dict() -> None:
    cfg = NotifyConfig.from_dict({"enabled": False, "min_probability": 0.8})
    assert cfg.enabled is False
    assert cfg.min_probability == 0.8


def test_notifier_skips_when_disabled() -> None:
    notifier = Notifier(NotifyConfig(enabled=False))
    # 不应抛出异常
    notifier.maybe_notify("title", "msg", probability=0.9)


def test_notifier_skips_below_threshold() -> None:
    notifier = Notifier(NotifyConfig(min_probability=0.8))
    # 不应抛出异常
    notifier.maybe_notify("title", "msg", probability=0.7)
