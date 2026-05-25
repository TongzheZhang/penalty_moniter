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

