"""D6 false-positive filtering.

STUB. Current behaviour: only the PDF's eligibility rule is applied (traced path length at least
MIN_TRACE_MM, which is also the primary test of D6 rule 2). Rules 1 (cropped ends), 3 (parallel
vessels), 4 (bowel/bone), 6 (duplicates) and 7 (origin diameter under MIN_ORIGIN_DIAMETER_MM
measured perpendicular to the path just outside the mask, borderline_diameter flag for
BORDERLINE_ORIGIN_DIAMETER_MM, wall patches under MIN_WALL_PATCH_VOXELS) per SPEC.md D6 are added
here one function each. The rejection log format is final: (label, rule name, value); it is the
tuning tool and the "known failure cases" slide.

Output contract (SPEC.md D9): surviving labels, rejection log of (label, rule, value).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import config
from candidates import Candidates
from instances import Instances

log = logging.getLogger("branchseed.filters")


@dataclass
class FilterResult:
    kept: list = field(default_factory=list)          # labels that survive, in instance order
    rejections: list = field(default_factory=list)    # (label, rule, value)


def rule_min_trace(trace) -> tuple[bool, float]:
    """PDF eligibility / D6 rule 2 primary: followable for at least MIN_TRACE_MM beyond the wall."""
    return trace.path_length_mm >= config.MIN_TRACE_MM, float(trace.path_length_mm)


def apply(cand: Candidates, inst: Instances, ostia: dict, traces: dict, frame=None) -> FilterResult:
    res = FilterResult()
    for label in inst.labels_list:
        tr = traces.get(label)
        if tr is None or label not in ostia:
            res.rejections.append((label, "no_trace", 0.0))
            continue
        ok, value = rule_min_trace(tr)
        if not ok:
            res.rejections.append((label, "min_trace_mm", value))
            continue
        res.kept.append(label)
    for label, rule, value in res.rejections:
        log.info("reject label %d: %s = %.2f", label, rule, value)
    log.info("%d kept, %d rejected", len(res.kept), len(res.rejections))
    return res
