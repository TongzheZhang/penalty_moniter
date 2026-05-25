import json

from src import tuning
from src.config import Settings


def test_grid_search_finds_best_threshold(tmp_path) -> None:
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

    settings = Settings()
    results = tuning.run_grid_search(
        input_path,
        thresholds=[0.70, 0.80, 0.90],
        confidences=[0.50, 0.60],
        base_settings=settings,
    )

    assert len(results) == 6
    # threshold=0.8 应该能过滤掉 evt_false（概率低），保留 evt_true（概率约 0.82）
    r_08 = [r for r in results if r.probability_threshold == 0.80]
    for r in r_08:
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.false_positive == 0

    # threshold=0.9 时 evt_true 也低于阈值，没有正例预测
    r_09 = [r for r in results if r.probability_threshold == 0.90]
    for r in r_09:
        assert r.precision is None
        assert r.recall == 0.0


def test_tuning_report_format() -> None:
    results = [
        tuning.TuneResult(
            probability_threshold=0.8,
            min_confidence=0.5,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            total_pnl=10.0,
            paper_orders=1,
            true_positive=1,
            false_positive=0,
            true_negative=1,
            false_negative=0,
            evolution_candidates=0,
        )
    ]
    report = tuning.format_tuning_report(results, top_k=1)
    assert "100.00%" in report
    assert "1.0000" in report


def test_save_tuning_results(tmp_path) -> None:
    results = [
        tuning.TuneResult(
            probability_threshold=0.75,
            min_confidence=0.55,
            precision=0.5,
            recall=1.0,
            f1=0.6667,
            total_pnl=5.0,
            paper_orders=2,
            true_positive=1,
            false_positive=1,
            true_negative=0,
            false_negative=0,
            evolution_candidates=1,
        )
    ]
    path = tmp_path / "tune.json"
    tuning.save_tuning_results(results, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["results"][0]["probability_threshold"] == 0.75
