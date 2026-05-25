from src.models import EvidenceEvent, SignalScores


def test_signal_scores_clamp_values() -> None:
    scores = SignalScores.from_dict({"box_contact_score": 2, "fall_score": -1})
    assert scores.box_contact_score == 1.0
    assert scores.fall_score == 0.0


def test_evidence_event_roundtrip() -> None:
    event = EvidenceEvent.from_dict(
        {
            "event_id": "evt1",
            "match_id": "match1",
            "source": "manual",
            "signals": {"box_contact_score": 0.5},
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.4},
            "match_context": {"home": "A", "away": "B", "minute": 50},
        }
    )
    payload = event.to_dict()
    assert payload["event_id"] == "evt1"
    assert payload["signals"]["box_contact_score"] == 0.5
    assert payload["market_snapshot"]["market_id"] == "m1"
    assert payload["match_context"]["minute"] == 50

