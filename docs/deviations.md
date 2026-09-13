# Deviations from the implementation spec

`docs/implementation_spec.md` is the build contract and `docs/design.md` the
rationale behind it. Everything below departs from one of them. Each entry says
what changed, why, and what evidence drove it, so the decision can be reversed
by someone who disagrees.

---

## 1. The centreline is extended to the cut faces

**Spec:** Stage 3 fits a spline to the skeleton's longest path and resamples it
at 1 mm.

**Built:** the same, then extrapolated along each end tangent, in 1 mm steps,
while the point remains inside the mask (capped at
`geometry.max_extension_mm`, 25 mm).

**Why:** a 3D skeleton retracts from a flat cut face. Measured on the synthetic
suite, the fitted centreline stopped 6 mm short at *each* end of a 180 mm tube.
Stage 4 casts rays perpendicular to the tangent, so those 6 mm of wall were
reachable by no ray at all — and an origin a few millimetres from a cut face is
a case the challenge calls out explicitly. Extension restores coverage to the
terminal plane, where the end-cap tests then take over.

---

## 2. The radius profile is stabilised near the ends

**Spec:** `radius_mm[i]` is the EDT interpolated at each centreline sample.

**Built:** the same, then each sample within `end_radius_window_mm` (12 mm) of
an end is raised to the largest radius seen between it and the interior.

**Why:** a direct consequence of deviation 1. The EDT measures distance to the
nearest background voxel, and within a voxel of a terminal plane that
background *is* the plane, so the profile collapsed to 1.1 mm at the endpoint
of a 9 mm tube. Radius drives the ray-length cap and both end-cap windows, so
the collapse disabled the plane test exactly where it is needed: end-cap
exclusion fell from 235 to 8 voxels. A vessel does not change calibre abruptly
over a centimetre, so a one-sided running maximum is a safe repair and leaves
any genuine interior taper untouched.

An earlier attempt recomputed the EDT on a mask continued past each terminal
plane. It was more principled and less robust — the extension endpoint can sit
up to half a voxel outside the true face, which left a sliver of unfilled
background — and it cost an extra full-volume distance transform. Dropped in
favour of the simpler mechanism.

---

## 3. The trace propagates at a permissive threshold

**Spec:** `lumen = (ct > t_lumen) & (ct < t_calcium) & ~aorta_mask`, and the
weak-match test asks whether fewer than half the band's voxels exceed
`t_lumen`.

**Built:** propagation runs at `t_lumen + trace.propagate_sigma_offset ·
sigma_ao`, with the offset at −2.0; the strong-match test still runs at the
strict `t_lumen`.

**Why:** as literally specified the weak-match test is vacuous — if the
propagation region is defined by `ct > t_lumen`, then 100% of every band
exceeds `t_lumen` by construction, and the criterion can never fire.
Separating the two thresholds makes it mean something. It also matters
physically: a 2 mm-radius vessel at 1.5 mm spacing is partial-volume dimmed
below the aortic lumen value, so tracing at `t_lumen` would kill precisely the
small branches that are hardest to recall.

---

## 4. The bifurcation test and the radius diagnostic are gated on distance

**Spec:** evaluate the stop conditions at every level from 0.5 mm.

**Built:** the bifurcation test and the radius-drop diagnostic only run beyond
`trace.stable_from_mm` (2.0 mm). Leak, weak match and the 10 mm cap are
unchanged.

**Why:** the geodesic source is a curved patch on the aortic surface, so for
the first millimetre or two the wavefront is still adapting to the shape of
that patch rather than to the branch lumen. Measured: a plain single-tube
branch registered a "bifurcation" at 1.0 mm and picked up a spurious
`seed_beyond_bifurcation` flag, and `radius_drop` fired on nearly every
candidate because the first level's band is inflated by the spread of the
patch. Both are artefacts of the source geometry, not of the vessel.

---

## 5. `detect.min_region_px` lowered from 4 to 2

**Spec:** 4.

**Why:** measured on the synthetic suite, a 2 mm-radius branch produces a clean
495 HU peak against a −50 HU background but clears the absolute threshold on
only three pixels, because suprathreshold *area* scales with branch calibre.
At 4 it was the only missed branch in the suite. Rejecting on area penalises
exactly the small accessory branches the challenge cares about, and the
design's own philosophy is to generate candidates generously in Stage 4 and
reject in Stages 5 and 7 — where the tests are principled rather than a pixel
count. Recorded in the README's parameter table as phantom-tuned and subject to
re-fitting on real data.

---

## 6. The bump rule uses followed length, not truncated length

**Spec:** plot `path_length_mm` against `path_length_mm / local_aortic_radius`.

**Built:** the feature is `max(reached_mm, path_length_mm)` — how far the lumen
was followable — with `path_length_mm` still reported as the honest proximal
length.

**Why:** `path_length_mm` is truncated at a bifurcation. Scoring a trunk that
divides at 5.5 mm as a 5.5 mm stub let this rule reject it as an artefact
(measured score −0.046, just the wrong side of the line) even though it had
already passed the eligibility test. A bifurcation is evidence *for* a real
vessel; penalising it double-counts and lets a fitted-by-hand discriminant undo
the substantive rule.

The boundary itself was also re-placed, from `[0.6, 4.0, −6.0]` to
`[0.4, 10.0, −7.0]`, weighting the relative axis more heavily and moving it
clear of legitimate cases. It remains **unfitted** — see the README.

---

## 7. scikit-fmm is not used

**Spec:** fast marching preferred, Dijkstra as the fallback — with a note
recommending Dijkstra anyway.

**Built:** Dijkstra only, on the 26-connectivity voxel graph.

**Why:** we took the spec's own recommendation. One fewer wheel to vendor for
an offline install is worth more than the few-percent lattice bias on a
diagonal over a 10 mm budget. Measured trace cost is ~0.13 s per candidate.

The bias is visible and worth recording: the level-set equivalent radius decays
from 3.00 mm at d = 1 mm to 2.39 mm at d ≥ 5 mm on a branch whose region radius
is 3.05 mm, because the wavefront is not planar. It saturates, so the
ratio-based leak test is unaffected, and the *reported* radius comes from the
FWHM cross-section rather than from here.

---

## 8. `direction_xyz` is rounded to 6 decimals, not 3

**Spec:** "round to 3 decimal places on output" and, in the test list,
"`direction_xyz` is unit to 1e-6". These conflict: a unit vector at 3 decimals
is only unit to about 1e-3.

**Built:** coordinates at 3 decimals, direction at 6. Both intents are met and
the test asserts the 1e-6 bound.

---

## 9. `build_wall_map` shares the pipeline's spline prefilter

**Spec:** one batched `map_coordinates` call over the whole map.

**Built:** batched in blocks of 48 centreline samples, with the cubic prefilter
computed once for the ROI and shared with the trace, the seed cross-sections
and the contact probes.

**Why:** a single call over the whole map allocates a coordinate array that
grows without bound with the length of the supplied segment. Blocking keeps
peak memory flat in segment length, and sharing the prefilter removed a
duplicate pass. Measured: peak RSS 681 MB to 504 MB on a 2.36 M-voxel ROI, with
no change in runtime. The prohibition the spec actually cares about — per-point
`map_coordinates` calls in a Python loop — is respected: there are none
anywhere in the package.

---

## 10. Not built

* **Vesselness fusion (C.4.4).** Optional in the spec, and gated there on
  measuring an F1 improvement first. The intensity map alone saturates the
  synthetic suite, so there is nothing to measure against; it would cost the
  ~20 s the design names as the first thing to cut. Revisit only if real-data
  F1 plateaus.
* **`resolve.merge_by_contiguity` as an active step.** Stage 4 already settles
  contiguity, with circular padding and a watershed split for separated
  maxima. The function exists, documents the rule, and asserts the invariant,
  but performs no merge — deliberately, because the important half of that
  rule is the merge it forbids.
