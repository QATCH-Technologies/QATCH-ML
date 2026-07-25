from src.systems.qmodel_7_onyx.qa.analyze_replay import summarize_machine


def _rec(final_correct, false_forward=0, backward=0, latencies=None):
    return dict(
        scores={
            "ordinal": dict(
                latencies=latencies or {},
                false_forward=false_forward,
                backward=backward,
                final_correct=final_correct,
            )
        }
    )


def test_summarize_machine_counts_failing_runs():
    per_run = {
        "00001": _rec(final_correct=True),
        "00002": _rec(final_correct=False),
        "00003": _rec(final_correct=False),
    }
    s = summarize_machine("ordinal", per_run)
    assert s["failing"] == ["00002", "00003"]


def test_summarize_machine_missed_confirmation_when_latency_is_none():
    per_run = {
        "00001": _rec(final_correct=True, latencies={"1ch": 2.0, "2ch": None}),
    }
    s = summarize_machine("ordinal", per_run)
    assert s["missed_by_state"] == {"2ch": 1}
    assert s["lat_by_state"]["1ch"].tolist() == [2.0]


def test_summarize_machine_tracks_false_forward_and_backward_runs():
    per_run = {
        "00001": _rec(final_correct=True, false_forward=1),
        "00002": _rec(final_correct=True, backward=2),
        "00003": _rec(final_correct=True),
    }
    s = summarize_machine("ordinal", per_run)
    assert s["false_fwd_runs"] == ["00001"]
    assert s["backward_runs"] == ["00002"]


def test_summarize_machine_ignores_runs_without_this_machine():
    legacy_score = dict(latencies={}, false_forward=0, backward=0, final_correct=True)
    per_run = {"00001": dict(scores={"legacy": legacy_score})}
    s = summarize_machine("ordinal", per_run)
    assert s["failing"] == []
    assert s["missed_by_state"] == {}
