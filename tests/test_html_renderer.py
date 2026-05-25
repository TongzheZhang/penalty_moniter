from src.reporting.html_renderer import render_analysis_report


def test_render_analysis_report_contains_cards() -> None:
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
    html = render_analysis_report(results)
    assert "<!DOCTYPE html>" in html
    assert "事件总数" in html
    assert "100.00%" in html
    assert "box_contact_high" in html
    assert "bar-tp" in html
