from src.online.signal_emitter import SignalEmitter, ModelSignal


def test_signal_emitter_basic() -> None:
    emitter = SignalEmitter(probability_threshold=0.75, cooldown_sec=60.0)

    class FakeResult:
        probability = 0.8
        confidence = 0.7
        seconds_to_event = 15.0

    signal = emitter.emit(FakeResult(), 0.0)
    assert signal.triggered is True
    assert signal.smoothed_probability > 0.7


def test_signal_emitter_cooldown() -> None:
    emitter = SignalEmitter(probability_threshold=0.75, cooldown_sec=60.0)

    class FakeResult:
        probability = 0.8
        confidence = 0.7
        seconds_to_event = 15.0

    signal1 = emitter.emit(FakeResult(), 0.0)
    assert signal1.triggered is True

    # 冷却期内再次触发
    signal2 = emitter.emit(FakeResult(), 1.0)
    assert signal2.triggered is False


def test_signal_emitter_no_rise_edge() -> None:
    emitter = SignalEmitter(probability_threshold=0.75, rise_edge_min_delta=0.3)

    class FakeResult:
        probability = 0.76
        confidence = 0.7
        seconds_to_event = 10.0

    # 连续高概率，没有上升沿，不应触发
    signal1 = emitter.emit(FakeResult(), 0.0)
    signal2 = emitter.emit(FakeResult(), 1.0)
    assert signal2.triggered is False


def test_signal_emitter_reset() -> None:
    emitter = SignalEmitter()
    emitter._prob_history.append(0.5)
    emitter._last_trigger_time = 100.0
    emitter.reset()
    assert len(emitter._prob_history) == 0
    assert emitter._last_trigger_time is None
