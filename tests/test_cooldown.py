from src.cooldown import CooldownTracker


def test_cooldown_blocks_within_window() -> None:
    tracker = CooldownTracker(cooldown_sec=30.0)
    assert not tracker.is_in_cooldown("match1", "2026-05-08T12:00:00Z")
    tracker.record("match1", "2026-05-08T12:00:00Z")
    assert tracker.is_in_cooldown("match1", "2026-05-08T12:00:10Z")
    assert tracker.is_in_cooldown("match1", "2026-05-08T12:00:29Z")
    assert not tracker.is_in_cooldown("match1", "2026-05-08T12:00:31Z")


def test_cooldown_different_matches_independent() -> None:
    tracker = CooldownTracker(cooldown_sec=30.0)
    tracker.record("match1", "2026-05-08T12:00:00Z")
    assert not tracker.is_in_cooldown("match2", "2026-05-08T12:00:05Z")


def test_cooldown_reset() -> None:
    tracker = CooldownTracker(cooldown_sec=30.0)
    tracker.record("match1", "2026-05-08T12:00:00Z")
    tracker.reset()
    assert not tracker.is_in_cooldown("match1", "2026-05-08T12:00:05Z")


def test_cooldown_integration_with_paper_execution() -> None:
    from src.agents.decision import DecisionAgent
    from src.agents.paper_execution import PaperExecutionAgent
    from src.config import Settings
    from src.models import EvidenceEvent

    settings = Settings()
    tracker = CooldownTracker(cooldown_sec=60.0)
    event = EvidenceEvent.from_dict(
        {
            "event_id": "evt1",
            "match_id": "match1",
            "timestamp_utc": "2026-05-08T12:00:00Z",
            "source": "manual",
            "frame_paths": ["f1.jpg"],
            "signals": {
                "box_contact_score": 0.9,
                "fall_score": 0.8,
                "protest_score": 0.7,
                "ref_earpiece_score": 0.9,
                "ref_var_walk_score": 0.8,
                "whistle_or_stoppage_score": 0.7,
            },
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.6},
            "match_context": {"score_home": 1, "score_away": 1, "minute": 70, "attacking_side": "home"},
        }
    )
    prediction = DecisionAgent(settings).predict(event)

    agent = PaperExecutionAgent(settings, cooldown=tracker)
    order1, blocks1 = agent.maybe_create_order(event, prediction)
    assert order1 is not None
    assert "cooldown_active" not in blocks1

    # 同一比赛 10 秒后再次触发，应被冷却期阻止
    event2 = EvidenceEvent.from_dict(
        {
            "event_id": "evt2",
            "match_id": "match1",
            "timestamp_utc": "2026-05-08T12:00:10Z",
            "source": "manual",
            "frame_paths": ["f2.jpg"],
            "signals": {
                "box_contact_score": 0.9,
                "fall_score": 0.8,
                "protest_score": 0.7,
                "ref_earpiece_score": 0.9,
                "ref_var_walk_score": 0.8,
                "whistle_or_stoppage_score": 0.7,
            },
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.6},
            "match_context": {"score_home": 1, "score_away": 1, "minute": 70, "attacking_side": "home"},
        }
    )
    order2, blocks2 = agent.maybe_create_order(event2, prediction)
    assert order2 is None
    assert "cooldown_active" in blocks2
