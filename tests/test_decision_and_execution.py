from src.agents.decision import DecisionAgent
from src.agents.paper_execution import PaperExecutionAgent
from src.config import Settings
from src.models import EvidenceEvent


def test_decision_high_cluster_becomes_actionable() -> None:
    settings = Settings()
    event = EvidenceEvent.from_dict(
        {
            "event_id": "evt1",
            "match_id": "match1",
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

    assert prediction.penalty_probability >= settings.probability_threshold
    assert prediction.confidence >= settings.min_confidence
    assert "ref_var_walk_high" in prediction.reason_codes
    assert prediction.side == "home"


def test_paper_execution_blocks_unmapped_market() -> None:
    settings = Settings()
    event = EvidenceEvent.from_dict(
        {
            "event_id": "evt1",
            "match_id": "match1",
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
            "market_snapshot": {"best_ask": 0.6},
            "match_context": {"score_home": 1, "score_away": 1, "minute": 70},
        }
    )
    prediction = DecisionAgent(settings).predict(event)
    order, blocks = PaperExecutionAgent(settings).maybe_create_order(event, prediction)

    assert order is None
    assert "market_or_token_unmapped" in blocks


def test_commentary_boosts_probability() -> None:
    settings = Settings()
    base_event = EvidenceEvent.from_dict(
        {
            "event_id": "evt1",
            "match_id": "match1",
            "source": "manual",
            "frame_paths": ["f1.jpg"],
            "signals": {
                "box_contact_score": 0.3,
                "fall_score": 0.2,
                "protest_score": 0.0,
                "ref_earpiece_score": 0.0,
                "ref_var_walk_score": 0.0,
                "whistle_or_stoppage_score": 0.0,
            },
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.6},
            "match_context": {"score_home": 1, "score_away": 1, "minute": 70, "attacking_side": "home"},
        }
    )
    base_pred = DecisionAgent(settings).predict(base_event)

    boosted_event = EvidenceEvent.from_dict(
        {
            "event_id": "evt2",
            "match_id": "match1",
            "source": "manual",
            "frame_paths": ["f1.jpg"],
            "signals": {
                "box_contact_score": 0.3,
                "fall_score": 0.2,
                "protest_score": 0.0,
                "ref_earpiece_score": 0.0,
                "ref_var_walk_score": 0.0,
                "whistle_or_stoppage_score": 0.0,
                "commentary_score": 0.9,
                "commentary_triggered": True,
            },
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.6},
            "match_context": {"score_home": 1, "score_away": 1, "minute": 70, "attacking_side": "home"},
        }
    )
    boosted_pred = DecisionAgent(settings).predict(boosted_event)

    assert boosted_pred.penalty_probability > base_pred.penalty_probability
    assert "commentary_penalty_mentioned" in boosted_pred.reason_codes


def test_multimodal_confirm_boosts_confidence() -> None:
    settings = Settings()
    event = EvidenceEvent.from_dict(
        {
            "event_id": "evt1",
            "match_id": "match1",
            "source": "manual",
            "frame_paths": ["f1.jpg"],
            "signals": {
                "box_contact_score": 0.7,  # visual high
                "fall_score": 0.0,
                "protest_score": 0.0,
                "ref_earpiece_score": 0.0,
                "ref_var_walk_score": 0.0,
                "whistle_or_stoppage_score": 0.0,
                "commentary_score": 0.8,
                "commentary_triggered": True,
            },
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.6},
            "match_context": {"score_home": 1, "score_away": 1, "minute": 70},
        }
    )
    prediction = DecisionAgent(settings).predict(event)
    assert "multimodal_confirm" in prediction.reason_codes

