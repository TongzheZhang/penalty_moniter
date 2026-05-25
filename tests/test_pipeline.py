import json

from src.agents.audit import AuditEvolutionAgent
from src.agents.context import ContextAgent
from src.agents.decision import DecisionAgent
from src.agents.market_sensor import MarketSensorAgent
from src.agents.paper_execution import PaperExecutionAgent
from src.agents.vision_sensor import VisionSensorAgent
from src.config import Settings
from src.pipeline import PenaltyResearchPipeline
from src.storage.jsonl_store import RunStore


def build_pipeline(tmp_path):
    settings = Settings()
    return PenaltyResearchPipeline(
        settings=settings,
        vision_sensor=VisionSensorAgent(),
        market_sensor=MarketSensorAgent(),
        context_agent=ContextAgent(),
        decision_agent=DecisionAgent(settings),
        paper_execution_agent=PaperExecutionAgent(settings),
        audit_agent=AuditEvolutionAgent(probability_threshold=settings.probability_threshold),
        store=RunStore(tmp_path),
    )


def test_replay_writes_outputs_and_candidates(tmp_path) -> None:
    events = [
        {
            "event_id": "evt_true",
            "match_id": "match1",
            "source": "manual",
            "frame_paths": ["f1"],
            "signals": {
                "box_contact_score": 0.92,
                "fall_score": 0.86,
                "protest_score": 0.72,
                "ref_earpiece_score": 0.90,
                "ref_var_walk_score": 0.84,
                "whistle_or_stoppage_score": 0.70,
            },
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.61},
            "match_context": {"score_home": 1, "score_away": 1, "minute": 72},
            "actual_outcome": "penalty_awarded",
            "price_after_120s": 0.82,
        },
        {
            "event_id": "evt_false",
            "match_id": "match2",
            "source": "manual",
            "frame_paths": ["f1"],
            "signals": {
                "box_contact_score": 0.72,
                "fall_score": 0.77,
                "protest_score": 0.82,
                "ref_earpiece_score": 0.88,
                "ref_var_walk_score": 0.78,
                "whistle_or_stoppage_score": 0.68,
            },
            "market_snapshot": {"market_id": "m2", "token_id": "t2", "best_ask": 0.46},
            "match_context": {"score_home": 2, "score_away": 2, "minute": 64},
            "actual_outcome": "no_penalty",
            "price_after_120s": 0.43,
        },
    ]
    input_path = tmp_path / "events.json"
    input_path.write_text(json.dumps(events), encoding="utf-8")

    summary = build_pipeline(tmp_path / "run").run_replay(input_path)

    assert summary["total_events"] == 2
    assert summary["paper_orders"] == 2
    assert summary["true_positive"] == 1
    assert summary["false_positive"] == 1
    assert summary["evolution_candidates"] == 1
    assert (tmp_path / "run" / "summary.json").exists()
    assert len((tmp_path / "run" / "audit.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_unlabeled_events_are_not_counted_as_false_positive(tmp_path) -> None:
    events = [
        {
            "event_id": "evt_unknown",
            "match_id": "match1",
            "source": "manual",
            "frame_paths": ["f1"],
            "signals": {
                "box_contact_score": 0.92,
                "fall_score": 0.86,
                "protest_score": 0.72,
                "ref_earpiece_score": 0.90,
                "ref_var_walk_score": 0.84,
                "whistle_or_stoppage_score": 0.70,
            },
            "market_snapshot": {"market_id": "m1", "token_id": "t1", "best_ask": 0.61},
            "match_context": {"score_home": 1, "score_away": 1, "minute": 72},
        }
    ]
    input_path = tmp_path / "events.json"
    input_path.write_text(json.dumps(events), encoding="utf-8")

    summary = build_pipeline(tmp_path / "run").run_replay(input_path)

    assert summary["total_events"] == 1
    assert summary["labeled_events"] == 0
    assert summary["false_positive"] == 0
    assert summary["evolution_candidates"] == 0
