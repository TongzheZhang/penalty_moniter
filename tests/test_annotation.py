import json

from src import annotation


def test_load_unlabeled_from_replay(tmp_path) -> None:
    events = [
        {"event_id": "e1", "actual_outcome": "penalty_awarded"},
        {"event_id": "e2", "actual_outcome": "unknown"},
        {"event_id": "e3"},
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events), encoding="utf-8")

    unlabeled = annotation.load_unlabeled_events(path)
    assert len(unlabeled) == 2
    assert unlabeled[0]["event_id"] == "e2"
    assert unlabeled[1]["event_id"] == "e3"


def test_load_unlabeled_from_run_dir(tmp_path) -> None:
    evidence = [
        {"event_id": "e1"},
        {"event_id": "e2"},
    ]
    audit = [
        {"event_id": "e1", "actual_outcome": "penalty_awarded"},
        {"event_id": "e2", "actual_outcome": "unknown"},
    ]
    (tmp_path / "evidence.jsonl").write_text(
        "\n".join(json.dumps(e) for e in evidence) + "\n", encoding="utf-8"
    )
    (tmp_path / "audit.jsonl").write_text(
        "\n".join(json.dumps(a) for a in audit) + "\n", encoding="utf-8"
    )

    unlabeled = annotation.load_unlabeled_events(tmp_path)
    assert len(unlabeled) == 1
    assert unlabeled[0]["event_id"] == "e2"


def test_batch_annotate(tmp_path) -> None:
    events = [
        {"event_id": "e1", "actual_outcome": "unknown"},
        {"event_id": "e2", "actual_outcome": "unknown"},
    ]
    mapping = {"e1": "penalty_awarded", "e2": "no_penalty"}
    annotated = annotation.batch_annotate(events, mapping)
    assert len(annotated) == 2
    assert annotated[0]["actual_outcome"] == "penalty_awarded"
    assert annotated[1]["actual_outcome"] == "no_penalty"


def test_save_annotated_json(tmp_path) -> None:
    events = [{"event_id": "e1", "actual_outcome": "penalty_awarded"}]
    path = tmp_path / "out.json"
    annotation.save_annotated_events(events, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["actual_outcome"] == "penalty_awarded"


def test_save_annotated_jsonl(tmp_path) -> None:
    events = [{"event_id": "e1", "actual_outcome": "penalty_awarded"}]
    path = tmp_path / "out.jsonl"
    annotation.save_annotated_events(events, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_id"] == "e1"
