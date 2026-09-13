# Branchseed — Ideas Handoff

Every technique worth using, why it works, and a worked example. Numbers come from case 25 (1.5 mm isotropic, aortic arch to just above the iliac bifurcation, lumen mean 485 HU) unless stated.

Companion to `design.md` (what the system is) and `implementation_spec.md` (how to build it). This document is the ledger of *ideas* — the things that are non-obvious, the things that are easy to get subtly wrong, and the reasoning behind each choice.

---

# Part 1 — The five governing ideas

Everything else follows from these.

## 1.1 The mask is a calibration source, not just an anchor

**The idea.** The supplied aorta mask contains a few tens of thousands of voxels of known arterial lumen from *this patient*, *this scanner*, *this contrast bolus*. Every threshold in the system should be derived from it.

**Why it matters.** Elattar names as their own unvalidated limitation that accuracy was never tested against other institutions' blood Hounsfield values. That is exactly the hidden-test-set risk. A fixed threshold of, say, 200 HU is a bet on contrast protocol.

**Worked example.** Case 25, eroded interior (EDT > 2 mm), 40,285 voxels:

```
mu_ao     = 485 HU
sigma_ao  =  32 HU
p5        = 433 HU   -> t_lumen
p50       = 486 HU
mu + 4σ   = 613 HU   -> t_calcium
```

A different case with a weaker bolus might give mu = 310, sigma = 28, t_lumen = 265. The same code produces the right threshold for both without touching a parameter.

**The guard.** If `sigma_ao / mu_ao > 0.25` the distribution isn't a clean contrast peak — thrombus in the mask, bad timing, or a leaked segmentation. Fall back to `mu - 1.5σ` (more permissive), set a `degenerate` flag, and carry it into the output and the display. Don't abort.

**Extension.** The same principle applies to the *background*: estimate it from the shell outside the mask, excluding anything above `t_lumen`. You need it for the FWHM half-maximum in §5.4.

## 1.2 Resolution is the binding constraint

**The idea.** At 1.5 mm isotropic, the 5 mm eligibility distance is 3.3 voxels and the 10 mm trace cap is 6.7. Every measurement must be sub-voxel or it quantises into noise.

**Worked example.** A renal artery of true radius 2.4 mm. Voxel-counted on a binarised mask at 1.5 mm, the cross-section is 5 or 8 voxels depending on where the grid falls, giving `sqrt(5·2.25/π) = 1.89 mm` or `sqrt(8·2.25/π) = 2.39 mm`. A 25% swing from grid alignment alone. FWHM on a cross-section interpolated at 0.25 mm gives ~2.4 mm regardless of alignment.

**The consequence.** Three specific design choices fall out:

- Radius by FWHM on an interpolated plane, never by EDT on a binarised mask (§5.4)
- Geodesic level sets at 0.5 mm in an upsampled subvolume, never voxel-stepping at native spacing (§4.2)
- All probe sampling by `map_coordinates(order=3)`, never nearest-neighbour lookup

**Elattar's corroboration.** They found the right coronary ostium less accurately than the left and attributed it to the smaller vessel diameter producing larger partial volume effects — at 0.44–0.68 mm in-plane. Every branch you're chasing is in that regime and worse.

## 1.3 The trace is short, so tracking machinery is the wrong tool

**The idea.** Ten millimetres. Wala's optimised cylinder height was 15 mm — longer than your entire permitted trace. Their step was 3 mm, so you'd fit three cylinders. Their radius-ratio bifurcation test compares against 7.5 mm upstream, which over a 10 mm budget has almost nothing to compare against.

**What to do instead.** Upsample a 30 mm subvolume to 0.5 mm and walk geodesic level sets. Twenty samples across the budget instead of three.

**Worked example.** A branch that bifurcates at 6.5 mm:

| Approach | Samples before the split | Detection |
|---|---|---|
| Cylinder tracking, 3 mm step | 2 | radius ratio on 2 points — unusable |
| Geodesic levels, 0.5 mm | 13 | component splits at level 13, persists at 14 and 15 |

The level-set version detects the topological event directly. The cylinder version infers it from a scalar with two samples.

**What survives from Wala.** The leak detector and the strong-match criterion — both are single-step tests, not tracking loops. See §6.1.

## 1.4 Precision is the hard half

**The idea.** Finding bright things near the aorta is easy. Finding *only* the ones that are branches is the problem.

**Worked example.** Case 25, 6 mm shell around the mask, thresholded at 433 HU:

```
connected components:        106
components with >=5 voxels:   38
largest component:          1788 voxels
total bright shell voxels:  3689
```

The real branch count for a segment covering arch to infrarenal is maybe 10–20. So roughly half your ≥5-voxel components are noise, and the 1788-voxel one is almost certainly a cardiac chamber or pulmonary artery adjacent to the ascending aorta.

**The four defences, in order of value:**

1. **Leak detector** (§6.1) — kills the big blobs
2. **Geodesic 5 mm survival** (§4.3) — kills the short bumps
3. **Wall-contact brightness** (§6.3) — kills the abutting veins
4. **Length vs length/radius boundary** (§6.4) — kills the surface artefacts

With branch discovery at 45% and one-to-one matching where duplicates count as false positives, this is where the score lives.

## 1.5 Coverage is unpredictable

**The idea.** The brief says "abdominal aorta". Case 25's mask runs from the aortic arch down to just above the iliac bifurcation.

**What this breaks:**

```
z=150  ncomp=1  area=80    <- infrarenal
z=250  ncomp=1  area=151
z=300  ncomp=2  area=263   <- arch: one axial slice, two cross-sections
z=360  ncomp=2  area=651
z=370  ncomp=1  area=426
```

- **No per-slice reasoning.** An axial slice through the arch contains two disconnected aortic cross-sections.
- **No superior-inferior ordering.** Order instance IDs by arc length along the centreline instead.
- **No abdominal priors.** Riffaud's Table 1 distances (157.7 mm span, 35 mm CA/SMA/RA cluster, 9.5 mm inter-renal) describe a segment that may not be present.
- **Great-vessel origins are in scope.** Brachiocephalic, carotid, subclavian are direct aortic daughters.

**And the end caps are interior.** Mask occupies z = 147–375 of 421 slices, so neither cut face touches the volume boundary. First and last occupied slices have areas of 82 and 84 voxels — full-calibre flat discs, ~185 mm² each. A boundary-based exclusion catches nothing. See §3.3.

---

# Part 2 — Ideas borrowed from the literature

## 2.1 The cylindrical wall map (Elattar 2016)

**The original.** To find coronary ostia, they build an image of the mean intensity in the volume between the aortic surface and a dilated surface. Each pixel is the average along a cylinder starting at the surface, 2.5 mm long, 0.75 mm radius. Rows are MPR slices along the centreline, columns are angle around it. Ostia are local maxima.

**Why it's the right primitive.** It converts an open-set 3D search into a 2D peak-detection problem on an image whose axes are exactly the two coordinates that describe a branch origin: where along the vessel, and at what clock position. It is also, for free, the clinician display.

**The adaptation.** Three changes, all forced by the open-set problem:

| Elattar | Branchseed | Why |
|---|---|---|
| Probe 2.5 mm, r = 0.75 mm | Probe 1–6 mm, r = 1 mm | Must span the 5 mm eligibility distance; coarser data needs a wider probe |
| Gaussian weight at the distal extent, σ = 4 mm | None | They know where coronary ostia are. You don't. |
| Two 1D max projections | 2D local maxima + connected components | Projection only works with a known count at one level |

**The projection point is the subtle one.** Their Fig. 4 shows two 1D curves: project along one axis to get the angular peaks, along the other to get the axial position. That's valid because there are exactly two ostia at roughly the same level, so collapsing the axial axis leaves two clean peaks in θ. Your branches sit at many different levels — project and they smear into each other.

**Worked example of the failure.** Three branches at (arc 40 mm, θ = 0°), (arc 90 mm, θ = 180°), (arc 140 mm, θ = 90°). Project along arc length: peaks at 0°, 90°, 180° — three peaks, fine. Project along θ: peaks at 40, 90, 140 mm — three peaks, fine. But you have no way to pair them. Nine candidate combinations, three correct. With 2D detection you read both coordinates off each region directly.

**Starting at 1 mm, not 0.** At 1.5 mm spacing the wall voxel is a blend of lumen and surrounding tissue. Including it adds a roughly constant offset to every pixel, compressing dynamic range. Starting at 1 mm skips the partial-volume layer.

**Ending at 6 mm.** Past the 5 mm eligibility distance, so a stub that dies at 3 mm gives a lower mean than a real branch. The map pre-filters for eligibility.

**Also take: calcium masking.** They mask high intensities from calcification before segmenting. Calcium is brighter than contrast and sits exactly where you're looking. Drop any probe sample above `t_calcium` from the mean; if more than half are calcium, mark the pixel invalid rather than reporting a misleading low value.

**Also take: the multiply-to-fuse pattern.** Their hinge-point detector multiplies three 2D maps (Gaussian curvature, min inward intensity, max inward intensity) and thresholds the product at half its maximum. Multiplication demands all criteria simultaneously; averaging lets one strong signal carry a weak one. If you add vesselness and wall curvature layers, multiply them.

## 2.2 The PCA direction with sign fix (Riffaud 2022)

**The formula.** Given traced path points and an origin **b**, build A with rows (vᵢ − b), take the eigenvector **e** of AᵀA with the largest eigenvalue, then

```
d = sign(1ᵀAe) · e
```

**Why the sign term is essential.** An eigenvector's polarity is arbitrary — `eigh` returns whichever sign the algorithm lands on. `1ᵀAe` is the sum of projections of all path points onto **e**; it's positive when the path runs in the +**e** direction. Multiplying by its sign forces the vector to point away from the ostium into the branch.

**Worked example.** Path points at 1, 2, …, 10 mm along a true direction of (0, 0.6, 0.8).

```
eigh returns e = (0, -0.6, -0.8)     # wrong polarity, 50% of the time
1ᵀAe = -55                            # negative
sign(-55) · e = (0, 0.6, 0.8)         # corrected
```

Without the fix, roughly half your detections get a direction vector pointing back into the aorta. The angular error is 180°, which the quality metric will notice.

**Why PCA beats the obvious alternative.** An ostium-to-seed difference vector uses two points. With 20 path samples and 1.5 mm data, the least-squares fit is materially more stable — a single mis-centred level-set centroid shifts a two-point vector by several degrees and the PCA fit by a fraction of one.

**Do it in physical millimetres.** If you do the PCA in index space and the direction cosines aren't identity, normalisation hides the error entirely — you get a unit vector pointing in the wrong direction with no symptom. Convert path points to physical mm first and the problem is structurally impossible.

**Use the truncated path.** Riffaud computes both a global direction (whole branch) and a local one (within 3r of the origin, stopping at the first bifurcation or leaf). The spec asks for the *initial* direction, so use the proximal path only. Their 3r rule is also a useful sanity complement to the flat 10 mm: for a fat branch off a fat aorta, 10 mm may be too short to be stable.

## 2.3 The non-anatomic branch feature space (Riffaud 2022)

**The observation.** The aorta isn't a perfect cylinder. Surface bumps generate spurious branches in any skeleton or tree extraction. Their Fig. 12 plots branch length against length-over-radius and shows real arteries separating from artefacts almost linearly, with artefacts clustered tightly at the bottom-left.

**Take the feature space. Do not take the constants.** Their rule:

```
length < 3r   OR   length < r + 20 mm   ->  non-anatomic
```

On case 25 with a local radius of 9.1 mm, that second clause rejects anything under 29.1 mm. The eligibility threshold is 5 mm and the trace cap is 10 mm. Their rule would delete every single daughter you're allowed to detect.

**What to do.** Plot `path_length_mm` against `path_length_mm / local_aortic_radius_mm` for your dev-set candidates, colour by true/false, and fit a boundary. A hand-placed line or a linear discriminant is more defensible on a handful of cases than a learned model.

**Why relative length is the discriminative axis.** A 4 mm protrusion on a 15 mm-radius aorta is almost certainly a bump. The same 4 mm protrusion on a 6 mm-radius aorta is proportionally much more prominent and more likely real. Absolute millimetres don't separate these; the ratio does.

**Their robustness figure is worth knowing.** 185 non-anatomic branches across 239 segmentations, of which only 18 caused errors — 9.7%. The artefacts are common but mostly harmless if you have a filter. Yours will be more common still: they note that trees from lumen-only segmentations contain more spurious branches than lumen-plus-thrombus, and a non-anatomic branch was misread as a second renal in 12 cases versus 7. **You have lumen only.**

## 2.4 The leak detector (Wala 2011)

**The rule.** A leak is detected when a fitted cylinder has a radius over 150% of the radius of the cylinder one full cylinder-length upstream.

**Why this is the highest-value borrowed rule.** Real arteries taper. Anything that balloons is the trace escaping into a structure that isn't a vessel — in Wala's case the mediastinum, in yours a cardiac chamber, the IVC, or contrast-filled bowel. It's the defence against case 25's 1788-voxel component.

**Adapted form.** Over the geodesic level-set walk, with a 3 mm lookback:

```
if r(d) > 1.5 * r(d - 3mm):  leak
```

**Worked example.** Trace from an ostium at the ascending aorta that actually enters the right atrium:

```
d (mm):  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0   4.5
r (mm):  2.1  2.0  2.1  1.9  2.2  2.4  3.4  6.8  14.2
                                        ^
                              r(3.5)=3.4 vs r(0.5)=2.1 -> ratio 1.62 -> LEAK
```

Caught at 3.5 mm, before the candidate ever reaches the 5 mm eligibility threshold. Compare a real renal artery:

```
d (mm):  0.5  1.0  2.0  3.0  4.0  5.0  6.0  7.0
r (mm):  2.6  2.5  2.4  2.4  2.3  2.2  2.2  2.1     ratio never exceeds 1.0
```

**Discard, don't truncate.** A leak means the candidate was never a branch. Truncating it at the leak point would leave a plausible-looking 3.5 mm stub that then fails eligibility anyway — but if the leak happened at 7 mm you'd keep a false positive. Discard outright.

## 2.5 The asymmetric similarity penalty (Wala 2011)

**The original.** Cylinder-to-image similarity scores +1 per foreground voxel and **−5** per background voxel. They state the harsh penalty exists to stop the tracker jumping between adjacent soft-tissue structures.

**The transferable idea.** When testing whether a candidate region is a coherent vessel, punish gaps far more than you reward fill. A region that spans the gap between the aorta and a neighbouring bright structure contains a thin band of dark tissue; symmetric scoring lets the two bright ends outvote it.

**Where to use it.** In the strong-match test during tracing: require at least 50% of a level band's voxels above `t_lumen`, and weight the shortfall heavily. Also in candidate scoring if you build a fused map — a low value anywhere in the probe range should suppress the pixel, which is the same reasoning as multiplying rather than averaging (§2.1).

## 2.6 Progressive morphological opening (Wala 2011)

**The original.** To separate the pulmonary artery from the adjacent heart, open with a small kernel; if no candidate separates out, increase the kernel by one voxel and retry until it does.

**Why it's clever.** A fixed opening either fails to separate touching structures or destroys the small ones you care about. The smallest kernel that achieves separation preserves the most.

**Where to use it.** As a fallback when a candidate's wall-contact patch is suspiciously large — say, more than 40% of the local circumference. That usually means the aorta is touching another bright structure and the "branch" is the contact zone. Open progressively until the candidate either separates into a plausible branch or disappears.

## 2.7 False-positive bifurcation correction (Wala 2011)

**The original.** If child 1 terminates after only a few iterations, erase it and let child 2 continue as a continuation of the parent.

**The adaptation.** When a level-set split produces two sub-components and one of them dies within 1–2 mm, it wasn't a bifurcation — it was noise or a small surface irregularity. Merge back and continue the trace as if no split occurred.

**Why it matters here.** With a 10 mm budget, a spurious split at 4 mm would truncate your path to 4 mm, fail eligibility, and lose a real branch. This correction is worth more to you than it was to Wala, because you have so little path to lose.

## 2.8 Parameter optimisation methodology (Wala 2011)

**The pattern.** Three cases held out for training. A table of parameter / range / step / optimised value. One score, `TP + TN − FP − FN`, maximised. One parameter swept at a time.

**Their finding worth internalising.** Cylinder height had the largest effect, and both too-small and too-large values degraded performance sharply: short cylinders are noise-sensitive and fail to extract structure, long ones aren't sensitive enough to detect bifurcations.

**Expect the same shape from your probe range.** Plot the curve rather than taking the argmax. Pick a value in the middle of a flat region, not on a narrow peak — a narrow peak on 25 dev cases is overfitting.

## 2.9 Flagging instead of deciding (Riffaud 2022)

**The original.** When the lowest renal artery sat under 75 mm from the aortic bifurcation, a warning was raised asking for validation. Their Fig. 18 shows the separation is clean — most cases at 90–140 mm with a handful of obvious outliers.

**The posture, not the rule.** Their specific threshold is abdominal-anatomy-dependent and useless to you. What transfers is: emit uncertainty alongside the answer rather than committing silently.

**Your flags:**

```
"flags": ["near_cut_face"]          # ostium within 5 mm of an excluded end cap
"flags": ["close_pair"]             # another ostium within 3 mm
"flags": ["low_confidence"]         # composite score below threshold
"flags": ["seed_beyond_bifurcation"]# trace split before 5 mm
"flags": ["calcium_adjacent"]       # >30% of probe samples masked as calcium
"flags": ["degenerate_calibration"] # sigma/mu > 0.25
```

Surface them on the display. It's the cheapest route to "useful for clinicians" and it's honest.

---

# Part 3 — Geometry ideas

## 3.1 Skeleton longest path via double Dijkstra

**The idea.** Build a graph over skeleton voxels with physical edge weights. Run Dijkstra from any node to find the farthest node A; run it again from A to find the farthest node B. The A–B path is the centreline.

**Why.** This is the standard tree-diameter trick — exact on a tree, close enough on a near-tree. It handles the arch correctly because it makes no assumption about direction. A "take the topmost and bottommost voxel" heuristic fails when the vessel doubles back.

**Prune first.** Skeletons of real aortas have spurs from surface irregularity — the same phenomenon as §2.3. For each degree-1 endpoint, walk to the nearest junction; delete if the path is shorter than the local radius. Iterate 2–3 times.

**Worked example.** Case 25's mask is one connected component of 58,520 voxels with a z-extent of 342 mm. If your centreline comes out at 342 mm you've probably fitted a straight line through the arch; expect meaningfully longer, since the arch adds curvature.

## 3.2 Rotation-minimising frame — the one nobody warns you about

**The problem.** To unroll the wall you need a reference direction at each centreline sample, perpendicular to the tangent, varying smoothly with no twist.

**What fails, and how.**

*Fixed global vector projected into the normal plane.* `u = normalize(g − (g·t)t)` for some fixed **g**. When **t** becomes parallel to **g**, the projection has near-zero length and **u** becomes numerically arbitrary. On an arch, the tangent sweeps through nearly every orientation, so this *will* happen. The symptom is a band of the wall map where the columns are scrambled — and it looks like noise, not like a bug.

*Frenet frame.* Defined from the curvature vector, so it's undefined where curvature is zero — which is most of a straight aortic segment — and flips discontinuously at inflection points.

**What works.** The double-reflection method (Wang et al., ACM TOG 2008). Propagate the frame from sample *i* to *i+1* with two reflections. Stable, second-order accurate, about ten lines, no degenerate cases.

Sketch:

```
given  x_i, t_i, u_i  and  x_{i+1}, t_{i+1}:
  v1 = x_{i+1} - x_i
  c1 = v1·v1
  uL = u_i - (2/c1)(v1·u_i) v1        # first reflection
  tL = t_i - (2/c1)(v1·t_i) v1
  v2 = t_{i+1} - tL
  c2 = v2·v2
  u_{i+1} = uL - (2/c2)(v2·uL) v2     # second reflection
  v_{i+1} = t_{i+1} × u_{i+1}
```

**Then align to anterior once, globally.** After propagation, rotate the whole frame so θ = 0 points anterior at a mid-centreline sample. In LPS, anterior is −y. Now θ reads as a clock position with 12 o'clock anterior, which is what a surgeon expects.

Note in the display legend that clock position is defined in the local vessel frame, which diverges from the patient axial plane in the arch. That's the correct definition for stent planning anyway.

## 3.3 End-cap exclusion — three tests, all needed

**The problem.** The flat superior and inferior faces created by cropping are not branch origins, and they're the strongest false-positive generator in the system. Case 25's terminal slices are ~185 mm² full-calibre discs.

**Test 1 — volume boundary.** Any surface voxel within one voxel of a ROI face.

*On case 25 this clears nothing.* The mask occupies z = 147–375 of 421 slices. The cut faces are in the volume interior. If this were your only test you'd have zero protection.

**Test 2 — normal parallel to tangent.** Near each centreline endpoint (within 3× local radius of arc length), compute the angle between the surface voxel's outward normal and the endpoint-directed tangent. Clear if under 35°.

*Why it works.* On a cylindrical wall the outward normal is perpendicular to the tangent, ~90°. On a flat cut face it's parallel, ~0°. A real branch leaving near the end has a normal somewhere in between but rarely under 35° — branches leave roughly radially.

**Test 3 — terminal plane.** Clear any surface voxel within 3 mm of the plane through the centreline endpoint with the tangent as normal, and within 1.5× radius of the endpoint.

*Why all three.* Test 2 alone can miss a slightly oblique cut. Test 3 alone would clear a genuine branch that happens to sit near the end. Together they're conservative in the right direction, and the `near_cut_face` flag tells the reader when a detection was close to the line.

**Verify visually.** Render `eligible_wall` with excluded voxels coloured differently. This is the most common silent failure in the pipeline. A correctly-working exclusion shows as a band of invalid pixels at each end of the wall map with no peaks in it.

## 3.4 Ray validity at the arch

**The problem nobody expects.** On the inner curvature of a bend, normal planes from different centreline samples intersect. A ray cast outward from sample *i* can cross the wall belonging to sample *i+40*.

**Worked example.** Aortic arch with a radius of curvature of ~30 mm and a vessel radius of ~14 mm. On the inner side, the normal planes at samples 40 mm apart along the arch converge and cross well inside the region you're probing. Your wall map's arch rows get contributions from the wrong part of the vessel.

**The fix.** Build a `cKDTree` over the centreline points. After finding a wall crossing, query it: if the nearest centreline sample is more than 3 mm of arc length from sample *i*, mark the pixel invalid.

**Why you'd never catch this by eye.** The map still renders as a plausible image. It just has a region where the geometry is wrong, and the peaks in it are at wrong positions.

---

# Part 4 — Detection and tracing ideas

## 4.1 Batched interpolation

**The idea.** The wall map needs roughly `128 angles × 250 samples × 11 probe depths × 5 disc offsets ≈ 1.76 M` interpolated CT samples. Build the entire `(N, B, S, 5, 3)` coordinate array and make **one** `map_coordinates` call.

**Why it matters.** `map_coordinates` has substantial per-call overhead. A Python loop over 1.76 M points takes minutes; one batched call on the same points takes well under a second. This is the difference between hitting and blowing the 60 s budget.

**Same pattern everywhere.** Cross-section resampling, subvolume extraction, ray casting. Always assemble coordinates first, call once.

## 4.2 Geodesic distance, not Euclidean

**The idea.** Eligibility is "the lumen can be followed for at least 5 mm beyond the aortic wall". Followed — along the vessel, not as the crow flies.

**Worked example.** A branch that leaves the aorta and immediately curves 60°. Its lumen at 5 mm of path length sits 4.3 mm from the ostium in a straight line. A Euclidean test at 5 mm would look 0.7 mm too far and might land outside the lumen entirely; a geodesic test follows the path.

**Implementation choice.** Two options:

*Fast marching* (`skfmm.distance` with a masked array) — first-order accurate, isotropic.

*Dijkstra* on a 26-connected voxel graph with physical edge weights (`scipy.sparse.csgraph.dijkstra`) — has the classic lattice bias, roughly 5% overestimate on pure diagonals.

**Recommendation: Dijkstra.** On a 60³ subvolume it's fast, and over 10 mm a 5% bias is 0.5 mm — well inside your other error sources. Removing a vendored dependency from the offline install is worth more than that.

## 4.3 The 5 mm survival test as eligibility

**The idea.** Eligibility isn't a separate check bolted on afterwards. It falls directly out of the trace: a candidate is eligible if the geodesic walk reached 5 mm without leaking.

**Important distinction.** Eligibility is *independent of the stop reason*:

| Trace outcome | Eligible? |
|---|---|
| Bifurcated at 7 mm | Yes — 5 mm was reached |
| Reached 10 mm cap | Yes |
| Weak match died at 3 mm | No |
| Leaked at 6 mm | No — leak invalidates regardless |
| Bifurcated at 3 mm, lumen continues past 5 mm | Yes — one instance, follow dominant child |

That last row is the ambiguous case. See §5.3.

## 4.4 Bifurcation by component splitting

**The idea.** At level *d*, the retained component splits into two or more sub-components that remain separate at *d* + 0.5 and *d* + 1.0.

**Why this rather than a radius ratio.** The split *is* the bifurcation. A radius ratio is an inference about the same event from a noisy scalar. At 0.5 mm levels you have 20 samples and the topology is unambiguous; the radius estimate at each level has maybe 15% noise.

**Keep the persistence requirement.** Requiring separation to hold for two further levels suppresses single-level splits from noise. Without it, a partial-volume dropout truncates your trace and costs you the branch.

**Keep Wala's ratio as a diagnostic.** A sharp radius drop *without* a topological split usually means the trace slipped off the vessel onto a neighbour. Flag it, don't stop on it.

## 4.5 Circular padding on the map

**The idea.** The wall map is cyclic in θ. Pad circularly before any neighbourhood operation, then crop back.

**The failure without it.** A branch at θ ≈ 0° — anterior, so the SMA and celiac, the most reliably present branches — straddles the array edge and gets labelled as two components. Two instances where there should be one. Under one-to-one matching that's one true positive and one false positive: you lose twice.

## 4.6 Watershed for close pairs

**The idea.** If one map component contains two local maxima separated by more than ~4 mm in *physical* distance, split it with a watershed seeded at the maxima.

**Converting map distance to physical.** Vertical separation is `Δrow × 1 mm` of arc length. Horizontal is `Δcol × (2π/128) × R`, where R is the local wall radius. At R = 9 mm, one column is 0.44 mm; at R = 15 mm it's 0.74 mm. Compute properly rather than using pixel counts.

**Why it matters.** Riffaud's 239-case series gives a mean inter-renal origin distance of 9.5 mm with SD 5.1 — so pairs at 4–5 mm are well within the distribution. The spec explicitly requires two nearby origins to be returned as two instances when they're separate at the aortic wall. A naive connected-component labelling merges them.

---

# Part 5 — Measurement ideas

## 5.1 Ostium on the wall, not on the centreline

**The error to avoid.** Riffaud's branch location is the bifurcation node on the aortic centreline. That sits one aortic radius inside the wall.

**Worked example.** Case 25, median local radius 9.1 mm. Using a centreline node gives a systematic ~9 mm error on a metric worth 25% of the score. For scale, Elattar's automated coronary ostium detection achieved 2.0–2.4 mm against interobserver variation of 2.4–3.2 mm. You'd be four times worse than human disagreement, systematically, in a fixed direction.

**The construction.** Weighted centroid of the wall-patch points, weighted by map intensity (a proxy for how much lumen sits behind each wall pixel). Then project onto the surface: the centroid of points on a curved surface sits slightly inside, so bisect along the local outward normal until the signed EDT is zero. Five iterations gets you under 0.05 mm.

## 5.2 Seed recentring

**The idea.** Take the path point at 5 mm geodesic distance, then resample a plane perpendicular to the fitted direction and move the seed to the intensity-weighted centroid of the lumen cross-section.

**Why.** The 15% quality category checks whether the seed lies *on* the matched daughter. A level-set band centroid can sit off-lumen where the band is asymmetric — which happens whenever the branch is curving, which is most of the time.

**Worked example.** A branch curving at 40°. The 5 mm geodesic band is a curved shell; its 3D centroid sits toward the inside of the curve, potentially 0.8 mm off the lumen axis on a 2.2 mm-radius vessel. Recentring on a perpendicular cross-section fixes it.

**Fallback.** If the component containing the centre is empty — the seed fell off entirely — keep the un-recentred point and flag `seed_recentre_failed`. Don't silently emit a point you know is wrong.

## 5.3 The sub-5 mm bifurcation ambiguity

**The situation.** A trunk that splits at 3 mm. Per the spec it is eligible (5 mm of lumen exists), it is one instance (common trunk has one direct aortic origin even if it divides shortly afterwards), but the trace rule stopped at 3 mm and the seed is defined at 5 mm — past the split, where "the daughter path" is ambiguous.

**Three candidate resolutions:**

1. Extrapolate the seed along the fitted direction to 5 mm. Simple, but the seed may land between the two children, in tissue.
2. Follow the dominant (larger cross-section) child. The seed lands on real lumen.
3. Midpoint of the two children's centroids. Same problem as (1).

**Recommendation: dominant child.** It's what a human tracing a centreline does naturally, so it's the most likely match to the reference annotation.

**But flag it and say so.** Set `seed_beyond_bifurcation`, keep `path_length_mm` as the honest distance to the split, and state the assumption in the README and demo. If you can ask the organisers, ask. An explicit documented assumption reads far better than a silent guess, and this is exactly the kind of thing the reproducibility category rewards.

## 5.4 FWHM radius, not EDT

**The idea.** Resample a plane perpendicular to the direction at the seed, ±6 mm at 0.25 mm. Threshold at the half-maximum between the local lumen peak and local background. Take the connected region containing the centre. Report `sqrt(area/π)`.

**Why not EDT on a binarised mask.** Two problems: it inherits whatever bias your threshold has, and it quantises to the voxel grid. See the worked example in §1.2 — a 25% swing from grid alignment alone.

**The half-maximum construction:**

```
peak = max within 1 mm of centre           # e.g. 470 HU
bg   = median of annulus at 5-6 mm radius  # e.g. -40 HU  (excluding calcium)
half = (peak + bg) / 2                     # 215 HU
```

FWHM is the standard sub-voxel vessel-calibre estimator precisely because it's robust to the absolute intensity level — it adapts to the local contrast rather than using a global threshold.

**Sanity bounds.** Clamp to [0.4, 8.0] mm and flag anything outside. Also check `radius_mm < 0.5 × local_aortic_radius` — a "daughter" as wide as the parent is a leak that escaped the detector.

**Not an average along the path.** The spec asks for the local radius *at the seed*. Riffaud's mean-over-branch is the wrong quantity.

## 5.5 Confidence as a composite

**The idea.** Not required, but valuable for the display, the flags, and the tuning harness.

```
c_map    = clip((peak_value - t_lumen) / (mu_ao - t_lumen), 0, 1)
c_path   = min(path_length_mm / 10, 1)
c_taper  = clip(1 - abs(r_5mm - r_1mm) / r_1mm, 0, 1)
confidence = c_map * c_path * c_taper
```

Multiply rather than average, for the same reason as §2.1 — a candidate weak on any axis should score low.

**Don't threshold on it in the main pipeline.** That decision belongs in instance resolution where it can be tuned against F1. Use it to order the display, drive the `low_confidence` flag, and plot the PR curve.

---

# Part 6 — Instance resolution ideas

## 6.1 Contiguity, never proximity

**The rule.** Two candidates merge only if their wall-contact patches are contiguous on the eligible wall.

**Why proximity is catastrophic here.** The spec is explicit: two nearby origins must be returned as two instances when they're separate at the aortic wall. Riffaud gives a mean inter-renal origin distance of 9.5 mm, SD 5.1, so a substantial fraction sit under 5 mm apart.

**Worked example.** Merge radius of 5 mm. Paired renals with origins 4.2 mm apart:

```
without merge:  2 TP
with merge:     1 TP + 1 FN
```

Renals are a large fraction of the branches in an abdominal segment. This one rule choice could cost you 15–20% of recall.

**Implementation.** Do it on the map, not in 3D. Stage 4's connected-component labelling already encodes contiguity. So the rule reduces to: don't merge anything the labelling separated, and don't re-merge what the watershed split.

**The common trunk needs no rule.** The bifurcation stop already terminates the trace at the split while keeping exactly one ostium.

## 6.2 Daughter-of-daughter by claimed-voxel overlap

**The rule.** A vessel arising from another daughter is not a direct aortic daughter.

**Implementation.** Maintain a cumulative `claimed` boolean array. Process candidates in descending order of wall-patch area — the true parent should be established first. After accepting one, rasterise its traced lumen (dilated 1 mm) into `claimed`. For each subsequent candidate, if more than ~30% of its first 2 mm of trace overlaps `claimed`, reject.

**Why area ordering.** A secondary branch has a smaller wall footprint than its parent, so descending area processes parents first. Ties broken by confidence.

**Why the first 2 mm.** A genuine adjacent branch may run alongside a neighbour further out; what distinguishes a secondary branch is that its *origin* is on another vessel.

## 6.3 Wall-contact brightness for veins

**The idea.** A vein abutting the aorta looks connected in 3D, but there's no bright lumen *at the wall* between them.

**The test.** Median CT over the wall-patch voxels' outward 1–2 mm must exceed `t_lumen`. This is the first 1 mm of the probe, so you already have it from the map.

**Why it's better than an HU band alone.** In a well-timed CTA the arterial/venous HU gap does most of the work — case 25's arterial mean is 485, and a vein in the same scan might be 120. But timing varies, and a late-phase scan narrows the gap. The contact test is geometric rather than intensity-based and holds up better.

## 6.4 Order by arc length

**The idea.** Assign `branch_001`, `branch_002`, … by arc length along the centreline.

**Why not z.** Superior-to-inferior is ambiguous across an arch — case 25 has one. Two branches at the same z could be on opposite limbs. Arc length is well-defined for any coverage and makes IDs stable and reproducible across runs, which matters for the determinism test and for diffing your own outputs during development.

---

# Part 7 — Engineering ideas

## 7.1 Crop first, everything else follows

**The numbers.** Case 25: mask bbox 39 × 65 × 229 voxels out of 236 × 236 × 421 — about 2.5% of the volume. After 30 mm dilation you're at roughly 79 × 105 × 269 ≈ 2.2 M voxels.

**What this buys.** EDT, skeletonisation and any filtering run on 2.2 M voxels instead of 23 M. Peak memory sits near 1–2 GB against an 8 GB limit. And it's what makes the 60 s budget comfortable rather than tight.

**Do it immediately after load,** then `del` the uncropped arrays and let the SimpleITK image go out of scope. Convert int16 to float32 once, at crop time.

## 7.2 Don't globally upsample

**The idea.** Sub-voxel precision comes from `map_coordinates` at the points you need and from 30 mm per-candidate subvolumes — not from resampling the whole ROI.

**The arithmetic.** Upsampling a 2.2 M-voxel ROI to 0.5 mm gives ~60 M voxels, 240 MB per float32 array. Affordable, but the EDT and skeletonisation become an order of magnitude slower for no accuracy gain in those steps — they operate on the binary mask, where interpolation adds nothing.

**Where you do upsample.** The 30 mm subvolume per candidate: 60³ = 216 k voxels at 0.5 mm. Negligible. And the cross-section at the seed, at 0.25 mm. That's it.

## 7.3 Gzip sniffing

**The concrete finding.** The uploaded case-25 files are named `mask25.nii` and `orig25.nii`. `file` reports gzip compressed data. `nibabel.load` raises `ImageFileError: Cannot work out file type`.

**The fix.** Read the first two bytes; `1f 8b` means gzip regardless of extension. If extension and content disagree, copy to a temp path with the right extension and read that.

**Why it matters.** Five lines against a zero-score crash on the hidden set. And the hidden set came from the same source as case 25, so the same naming is likely.

## 7.4 Axis order — the highest-probability silent bug

**The hazard.** `sitk.GetArrayFromImage` returns `[z, y, x]`. `GetSize()`, `GetSpacing()`, `GetOrigin()` and `TransformContinuousIndexToPhysicalPoint` all use `(x, y, z)`.

**Why it's silent.** Mixing them produces output that looks entirely plausible — points land inside the body, directions are unit vectors, the JSON validates. It's just wrong, by a permutation.

**The discipline.** One convention, enforced by naming:

```
arr_zyx     # every numpy volume
pt_xyz      # every point, ROI continuous index
```

Any function that indexes an array from a point does the flip explicitly, in exactly one place (`interp.py`).

**And write the round-trip test first:**

```
p = ref.TransformIndexToPhysicalPoint((i, j, k))
q = ref.TransformPhysicalPointToContinuousIndex(p)
assert allclose(q, (i, j, k), atol=1e-6)
```

Run it as an assertion on every case, not just in the test suite.

## 7.5 LPS vs RAS

**The trap.** Nibabel reports RAS. SimpleITK reports LPS. The x and y signs differ. The spec mandates SimpleITK physical coordinates.

**Worked example.** Case 25's nibabel affine is diagonal-positive with origin (−176.654, −48.154, −552.1) in RAS. The same voxel in SimpleITK LPS has x and y negated. Emit RAS and every ostium is mirrored through two planes — every distance in the localisation metric is wrong, and nothing in your own testing would reveal it.

**The rule.** Prototype in whatever you like, but the emitting code path touches SimpleITK only. Never mix the two in one function.

## 7.6 Atomic writes and graceful failure

**Atomic write.** Write to `output.json.tmp`, then `os.replace`. A killed process never leaves a half-written file that looks valid.

**Graceful failure.** Wrap `process_case`:

- Optional stage fails (vesselness, figures) → catch, log, flag, continue
- Required stage fails → valid JSON, empty daughters list, `pipeline_error` flag, exception text in `meta`, **exit 0**
- Per-candidate failure → `rejected_by = 'exception'`, continue with the rest

**Why exit 0.** A crashed case scores zero on everything including reproducibility, and may abort the organisers' batch loop. An empty-list case scores zero only on discovery for that case and keeps the run alive.

## 7.7 Keep rejected candidates

**The idea.** Instance resolution sets `rejected_by` rather than deleting.

**Why.** When dev-set recall is disappointing, the single most actionable question is *which rule threw away the true positive*. With rejection reasons recorded, the evaluation harness answers it directly:

```
false negatives by cause:
  never_detected        3
  bump_filter           4     <- your boundary is too aggressive
  leak                  1
  secondary_branch      2     <- ordering problem?
```

Without it, you're re-running with rules disabled one at a time.

Also render them on the display as faint hollow markers with the reason in a legend. It turns the clinician figure into a debugging tool at no cost.

## 7.8 The offline install test

**The idea.** Build a container with networking disabled, run the README's setup command, then the run command on a dev case.

**Why.** The evaluation environment has no internet. This is exactly what the organisers will do, and it's where submissions most often fail — usually because `pip install` silently needed PyPI, or because a package downloads something on first import.

**The setup.** `pip download -r requirements.txt -d vendor/` on a connected machine, commit `vendor/`, and make the README command `pip install --no-index --find-links vendor -r requirements.txt`.

---

# Part 8 — The phantom suite

The highest-leverage thing to build early. Generates volumes with known ground truth, so you can test the whole pipeline before you have annotated dev data, and isolate algorithmic error from data messiness.

## 8.1 Generator

- Tube of radius 9 mm along a prescribed centreline, filled at 485 HU
- N side branches at prescribed arc lengths, clock angles, radii (1.5–4 mm), directions
- Additive Gaussian noise, σ = 30 HU
- Background at −50 HU, plus bright distractors *not* connected to the tube
- Rasterised at 1.5 mm isotropic with a **non-identity direction cosine matrix**
- Both ends flat-cut, positioned in the volume interior

That last detail on the direction cosines is deliberate — it's what catches the physical-vs-index-space errors in §5.1 and §7.5.

## 8.2 Variants, mapped to the spec's "Important cases"

| Phantom | Spec requirement | Assertion |
|---|---|---|
| Straight tube, 5 branches | baseline | 5 detected, 0 FP |
| Arched tube | varying coverage | 5 detected, frame doesn't twist |
| Two branches 8 mm apart | "two nearby origins → two instances" | 2 instances |
| Two branches 4 mm apart | same, harder | 2 instances (watershed fires) |
| Trunk splitting at 3 mm | "common trunk has one origin" | 1 instance, `seed_beyond_bifurcation` |
| Branch dying at 3 mm | eligibility rule | rejected |
| Branch opening into a 12 mm sphere at 6 mm | leak | rejected, `rejected_by='leak'` |
| Branch off a branch | "not a direct daughter" | 1 instance, secondary rejected |
| Branch at θ = 0° exactly | circular wrap | 1 instance, not 2 |
| Zero branches | "return an empty list" | `daughters: []`, exit 0 |
| Flat cut with a branch 4 mm from it | end caps + near-boundary | 1 instance, `near_cut_face` flag |

## 8.3 Accuracy assertions

- Ostium error under 2 mm (Elattar's benchmark was 2.0–2.4 mm at finer spacing)
- Direction angular error under 10°
- Radius error under 0.5 mm
- Output coordinates round-trip to the prescribed physical positions

## 8.4 Why this is worth the effort

Three reasons. It gives Claude Code (or you) a pass/fail signal to iterate against rather than eyeballing volumes. It works before any annotated data exists. And the variant table maps one-to-one onto the spec's "Important cases" list — being able to demonstrate each of them live is a strong two minutes of the five-minute demo.

---

# Part 9 — Calibration against the literature

| Source | Task | Result | How to use it |
|---|---|---|---|
| Elattar 2016 | Coronary ostium localisation, 0.44–0.68 mm in-plane | 2.37 ± 1.44 mm (RCA), 1.99 ± 1.30 mm (LCA); interobserver 2.38 ± 1.56 and 3.21 ± 4.89 | Target for the 25% localisation category. Adjust upward for 1.5 mm data. |
| Tahoces 2020 | Supra-aortic + visceral branches from a segmented aorta | recall 91.8%, precision 98.8% (F1 ≈ 0.95) | Closest comparable for the 45% category. They had a named list and 33 tuning cases — you have neither. |
| Riffaud 2022 | Named-artery ID from a pre-built vascular tree | 89.1% of 239 segmentations fully correct; polar renals only 70.8% | Ceiling for small branches even with a full segmentation handed to you. |
| Wala 2011 | Pulmonary arteries, low-dose CT, 1.25 mm slices | 64% sensitivity, 90% specificity; 0.15 mm bias, 0.63 mm RMS | Closest spacing to yours. Note localisation vastly exceeded detection. |

**The pattern across all four:** localisation is easier than detection. Once these systems find a vessel they measure it well; the failures are misses and confusions. Expect the same shape, and spend your effort on candidate generation and rejection rather than polishing geometry.

**Wala's stated tradeoff** is the one you'll face: relaxing the bifurcation model gives few missed segments but more false positives. Characterise that curve on your dev set, choose an operating point, and explain the choice. With one-to-one matching and duplicates counting as false positives, lean conservative.

---

# Part 10 — What has no precedent

Worth saying explicitly in the demo, because it's the part that's yours.

Every paper reviewed is handed either a pre-built vascular tree (Riffaud), a full segmentation including the branches (Riffaud, Tahoces), or a known fixed-count list of named targets at known locations (Elattar, Wala's seeding). None of them solves open-set branch discovery from a parent-lumen mask and raw CT.

So the following have no ancestor in the literature and had to be designed:

- Calibration from the eroded mask interior
- The geodesic 5 mm survival test as a direct implementation of the eligibility rule
- End-cap exclusion by normal-versus-tangent with interior cut faces
- The rotation-minimising frame for a wall map that traverses an arch
- Ray validity checking against centreline arc length
- Wall-patch-contiguity merging as the answer to "separate at the aortic wall"
- FWHM radius at a recentred seed
- Daughter-of-daughter rejection by claimed-voxel overlap
- LPS coordinate discipline and gzip sniffing

That list is roughly 80% of the work and essentially all of the risk. The literature gives you an ostium detector, a direction estimator, a leak test and a false-positive feature space — genuinely useful, worth citing, and not close to a solution.
