from src.storage.jsonl_store import JsonlStore, RunStore


def test_jsonl_store_roundtrip(tmp_path) -> None:
    store = JsonlStore(tmp_path / "test.jsonl", reset=True)
    store.append({"id": "a", "value": 1})
    store.append({"id": "b", "value": 2})
    rows = store.read_all()
    assert len(rows) == 2
    assert rows[0]["id"] == "a"
    assert rows[1]["value"] == 2


def test_jsonl_store_skips_corrupt_lines(tmp_path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text('{"id": "a"}\ninvalid json\n{"id": "b"}\n', encoding="utf-8")
    store = JsonlStore(path)
    rows = store.read_all()
    assert len(rows) == 2
    assert rows[0]["id"] == "a"
    assert rows[1]["id"] == "b"


def test_jsonl_store_read_missing_file(tmp_path) -> None:
    store = JsonlStore(tmp_path / "missing.jsonl")
    assert store.read_all() == []


def test_run_store_creates_files(tmp_path) -> None:
    store = RunStore(tmp_path / "run")
    assert (tmp_path / "run").is_dir()
    assert (tmp_path / "run" / "evidence.jsonl").exists()


def test_run_store_write_summary(tmp_path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_summary({"status": "ok", "events": 3})
    content = (tmp_path / "run" / "summary.json").read_text(encoding="utf-8")
    assert '"status": "ok"' in content
    assert '"events": 3' in content
