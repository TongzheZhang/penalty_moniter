import json

from src.batch_replay import generate_config_variants, run_batch_replay
from src.config import Settings


def test_generate_config_variants(tmp_path) -> None:
    base = {
        "decision": {"probability_threshold": 0.75},
        "paper": {"cooldown_sec": 0},
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")

    variants = generate_config_variants(
        base_path,
        {"decision.probability_threshold": [0.7, 0.8], "paper.cooldown_sec": [0, 30]},
    )

    assert len(variants) == 4
    ids = [v[0] for v in variants]
    assert "probability_threshold=0_7_cooldown_sec=0" in ids
    assert "probability_threshold=0_8_cooldown_sec=30" in ids


def test_batch_replay_runs_multiple_variants(tmp_path) -> None:
    events = [
        {
            "event_id": "evt_true",
            "match_id": "m1",
            "source": "manual",
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
            "match_id": "m2",
            "source": "manual",
            "signals": {
                "box_contact_score": 0.30,
                "fall_score": 0.20,
                "protest_score": 0.10,
                "ref_earpiece_score": 0.10,
                "ref_var_walk_score": 0.05,
                "whistle_or_stoppage_score": 0.10,
            },
            "market_snapshot": {"market_id": "m2", "token_id": "t2", "best_ask": 0.46},
            "match_context": {"score_home": 2, "score_away": 2, "minute": 64},
            "actual_outcome": "no_penalty",
            "price_after_120s": 0.43,
        },
    ]
    input_path = tmp_path / "events.json"
    input_path.write_text(json.dumps(events), encoding="utf-8")

    base_config = tmp_path / "config.json"
    base_config.write_text(json.dumps({"decision": {"probability_threshold": 0.75}}), encoding="utf-8")

    results = run_batch_replay(
        input_path,
        base_config,
        {"decision.probability_threshold": [0.75, 0.85]},
        tmp_path / "batch",
    )

    assert len(results) == 2
    # threshold=0.75 应该触发 1 个订单（evt_true），threshold=0.85 应该 0 个
    r75 = [r for r in results if r.variant_id == "probability_threshold=0_75"][0]
    r85 = [r for r in results if r.variant_id == "probability_threshold=0_85"][0]
    assert r75.summary["paper_orders"] == 1
    assert r85.summary["paper_orders"] == 0
