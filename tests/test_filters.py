import copy

import config
import filters


def test_phantom_branch_is_kept(cand, inst, ostia, traces):
    res = filters.apply(cand, inst, ostia, traces)
    assert res.kept == [1]
    assert res.rejections == []


def test_short_trace_is_rejected_with_logged_value(cand, inst, ostia, traces):
    short = copy.copy(traces[1])
    short.path_length_mm = config.MIN_TRACE_MM - 1.0
    short.seed_mm = None
    res = filters.apply(cand, inst, ostia, {1: short})
    assert res.kept == []
    assert len(res.rejections) == 1
    label, rule, value = res.rejections[0]
    assert (label, rule) == (1, "min_trace_mm")
    assert value == config.MIN_TRACE_MM - 1.0


def test_missing_trace_is_rejected(cand, inst, ostia):
    res = filters.apply(cand, inst, ostia, {})
    assert res.kept == [] and res.rejections[0][1] == "no_trace"
