from src.models import CommentaryTranscript, EvidenceEvent, SignalScores, VideoClip


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


def test_signal_scores_with_commentary() -> None:
    scores = SignalScores.from_dict(
        {"box_contact_score": 0.5, "commentary_score": 0.8, "commentary_triggered": True}
    )
    assert scores.commentary_score == 0.8
    assert scores.commentary_triggered is True
    assert scores.max_score() == 0.8
    payload = scores.to_dict()
    assert payload["commentary_score"] == 0.8
    assert payload["commentary_triggered"] is True


def test_commentary_transcript_roundtrip() -> None:
    ct = CommentaryTranscript(
        text="点球！",
        keywords_hit=["点球"],
        urgency_score=1.0,
        raw_confidence=0.9,
    )
    payload = ct.to_dict()
    assert payload["text"] == "点球！"
    assert payload["urgency_score"] == 1.0


def test_video_clip_roundtrip() -> None:
    clip = VideoClip(
        clip_path="clips/clip_001.mp4",
        start_ts=10.0,
        end_ts=20.0,
        trigger_event_id="evt1",
        trigger_probability=0.85,
    )
    payload = clip.to_dict()
    assert payload["clip_path"] == "clips/clip_001.mp4"
    assert payload["trigger_probability"] == 0.85

