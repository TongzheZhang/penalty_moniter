from src import analysis


def test_analyze_run_with_smoke_data() -> None:
    # data/runs/smoke 是项目自带的示例运行
    run_dir = __import__("pathlib").Path("data/runs/smoke")
    if not run_dir.exists():
        return

    result = analysis.analyze_run(run_dir)
    assert result["total_events"] == 3
    assert result["labeled_events"] == 3
    assert result["paper_orders"] == 2
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 0
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert "signal_analysis" in result
    assert "box_contact_score" in result["signal_analysis"]


def test_format_analysis_report() -> None:
    results = [
        {
            "run_dir": "/tmp/run1",
            "total_events": 2,
            "labeled_events": 2,
            "paper_orders": 1,
            "true_positive": 1,
            "false_positive": 0,
            "true_negative": 1,
            "false_negative": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "total_pnl": 5.0,
            "signal_analysis": {
                "box_contact_score": {"tp_mean": 0.9, "fp_mean": 0.0, "fn_mean": 0.0, "tn_mean": 0.3},
            },
            "reason_distribution": {"box_contact_high": 1},
            "failure_distribution": {},
        }
    ]
    report = analysis.format_analysis_report(results)
    assert "运行分析 report" in report
    assert "100.00%" in report
    assert "box_contact_high" in report
