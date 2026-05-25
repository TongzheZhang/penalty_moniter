from src.validation import validate_events, format_validation_report


def test_validate_clean_events() -> None:
    events = [
        {
            "event_id": "evt1",
            "match_id": "m1",
            "timestamp_utc": "2026-05-08T12:00:00Z",
            "signals": {
                "box_contact_score": 0.5,
                "fall_score": 0.3,
                "protest_score": 0.2,
                "ref_earpiece_score": 0.1,
                "ref_var_walk_score": 0.0,
                "whistle_or_stoppage_score": 0.0,
            },
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.5},
            "match_context": {"home": "A", "away": "B", "minute": 30, "attacking_side": "home"},
            "actual_outcome": "penalty_awarded",
        }
    ]
    issues = validate_events(events)
    assert not issues


def test_validate_detects_duplicate_event_id() -> None:
    events = [
        {"event_id": "evt1", "match_id": "m1", "timestamp_utc": "2026-05-08T12:00:00Z"},
        {"event_id": "evt1", "match_id": "m1", "timestamp_utc": "2026-05-08T12:01:00Z"},
    ]
    issues = validate_events(events)
    errors = [i for i in issues if i.severity == "error"]
    assert any("重复" in i.message for i in errors)


def test_validate_detects_signal_out_of_range() -> None:
    events = [
        {
            "event_id": "evt1",
            "match_id": "m1",
            "timestamp_utc": "2026-05-08T12:00:00Z",
            "signals": {"box_contact_score": 1.5, "fall_score": -0.1},
        }
    ]
    issues = validate_events(events)
    errors = [i for i in issues if i.severity == "error"]
    assert any("box_contact_score" in i.field for i in errors)
    assert any("fall_score" in i.field for i in errors)


def test_validate_detects_invalid_outcome() -> None:
    events = [
        {"event_id": "evt1", "match_id": "m1", "actual_outcome": "maybe"},
    ]
    issues = validate_events(events)
    warnings = [i for i in issues if i.severity == "warning"]
    assert any("maybe" in i.message for i in warnings)


def test_validate_detects_out_of_order_timestamps() -> None:
    events = [
        {"event_id": "evt1", "match_id": "m1", "timestamp_utc": "2026-05-08T12:02:00Z"},
        {"event_id": "evt2", "match_id": "m1", "timestamp_utc": "2026-05-08T12:01:00Z"},
    ]
    issues = validate_events(events)
    warnings = [i for i in issues if i.severity == "warning"]
    assert any("时间顺序" in i.message for i in warnings)


def test_format_report_with_issues() -> None:
    events = [
        {"event_id": "", "match_id": "m1"},
    ]
    issues = validate_events(events)
    report = format_validation_report(issues)
    assert "错误" in report
