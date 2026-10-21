"""Unit tests for eval.metrics.classification and refusal."""
from eval.parser import Pair
from eval.runner import Trace
from eval.metrics.classification import routing_report
from eval.metrics.refusal import refusal_report


def _pair(pid, expected_uas, must_cite=None, question="Q?"):
    return Pair(
        id=pid, tier="reference", bucket=expected_uas[0] if expected_uas[0] != "NONE" else "OOS",
        question=question, expected_ua_raw="|".join(expected_uas),
        expected_uas=expected_uas, must_cite=must_cite or [], tags=[],
        ground_truth="gt",
    )


def _trace(pid, predicted_ua, is_refusal=False):
    return Trace(
        pair_id=pid, tier="reference", bucket="UA-1",
        predicted_ua=predicted_ua, is_refusal=is_refusal,
        answer="…",
    )


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def test_routing_perfect():
    pairs = [_pair("p1", ["UA-1"]), _pair("p2", ["UA-2"])]
    traces = [_trace("p1", "UA-1"), _trace("p2", "UA-2")]
    r = routing_report(pairs, traces)
    assert r.accuracy == 1.0
    # weighted-F1 (by support) should be 1.0 when all predictions are correct;
    # macro-F1 over ALL_LABELS is pulled down by 9 zero-support classes — that's expected.
    assert r.weighted_f1 == 1.0
    assert r.per_class["UA-1"]["f1"] == 1.0
    assert r.per_class["UA-2"]["f1"] == 1.0

def test_routing_mistake():
    pairs = [_pair("p1", ["UA-1"]), _pair("p2", ["UA-2"])]
    traces = [_trace("p1", "UA-1"), _trace("p2", "UA-3")]
    r = routing_report(pairs, traces)
    assert r.accuracy == 0.5

def test_routing_multi_accept():
    # expected UA-1 OR UA-9 — predicting UA-9 must count as correct
    pairs = [_pair("p1", ["UA-1", "UA-9"])]
    traces = [_trace("p1", "UA-9")]
    r = routing_report(pairs, traces)
    assert r.accuracy == 1.0


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------

def test_refusal_precision_recall():
    pairs = [
        _pair("oos1", ["NONE"]),         # must refuse
        _pair("oos2", ["NONE"]),         # must refuse
        _pair("ok1", ["UA-1"]),          # must not refuse
    ]
    traces = [
        _trace("oos1", None, is_refusal=True),     # TP
        _trace("oos2", None, is_refusal=False),    # FN
        _trace("ok1", "UA-1", is_refusal=False),   # TN
    ]
    r = refusal_report(pairs, traces)
    assert r.true_positives == 1
    assert r.false_negatives == 1
    assert r.true_negatives == 1
    assert r.false_positives == 0
    assert r.precision == 1.0      # 1 TP / (1 TP + 0 FP)
    assert r.recall == 0.5         # 1 TP / (1 TP + 1 FN)

def test_refusal_false_positive():
    # agent refuses a valid question -> FP
    pairs = [_pair("ok", ["UA-1"])]
    traces = [_trace("ok", None, is_refusal=True)]
    r = refusal_report(pairs, traces)
    assert r.false_positives == 1
    assert r.precision == 0.0

def test_refusal_borderline():
    # borderline: NONE_OR_UA-5 — either behaviour is OK
    p = _pair("b1", ["NONE", "UA-5"])
    t_refuse = _trace("b1", None, is_refusal=True)
    r = refusal_report([p], [t_refuse])
    assert r.borderline_total == 1
    assert r.borderline_correct == 1
    # borderline isn't counted in the strict binary
    assert r.true_positives == 0 and r.false_negatives == 0
