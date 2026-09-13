"""Perturbation-stability check (SPEC D7): which predictions flip when the inputs move by the
amount the supplied masks and scans plausibly vary. Each case is run unperturbed (the baseline
pseudo-reference) and then with

  mask_erode1    the supplied mask eroded by one native voxel (a tighter segmentation)
  mask_dilate1   the supplied mask dilated by one native voxel (a looser one)
  thr_minus      the adaptive threshold lowered by THRESHOLD_SHIFT_HU (a fainter scan)
  thr_plus       the adaptive threshold raised by THRESHOLD_SHIFT_HU (a brighter one)

and the perturbed predictions are matched to the baseline with the scorer at the working cutoff
(coarse_check.compare_stacks: lost and gained branches explained by the fate of the wall patch
under them). On the labelled cases the reference matches are also re-counted per perturbation,
so a true positive that flips is named. A measurement, not a tuning set: nothing here changes a
constant.

    python perturb_check.py --out results/perturbation_check.txt [--cases 1-24]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import candidates
import coarse_check
import config
import io_utils
import scorer

THRESHOLD_SHIFT_HU = 10.0  # a few HU: the D1 threshold sits 50 to 300 HU under the lumen median, so 10 HU is a small fraction of its margin and about one noise std on the coarse cases
DEFAULT_CASES = "1-24"     # 25 (whole aorta with the arch) is out of scope
PERTURBATIONS = ("mask_erode1", "mask_dilate1", "thr_minus", "thr_plus")


def _morph_mask(mask_image: sitk.Image, mode: str) -> sitk.Image:
    arr = sitk.GetArrayViewFromImage(mask_image) > 0
    st = ndimage.generate_binary_structure(3, 1)  # 6-connected: one voxel along each axis
    out = ndimage.binary_erosion(arr, st) if mode == "erode" else ndimage.binary_dilation(arr, st)
    im = sitk.GetImageFromArray(out.astype(np.uint8))
    im.CopyInformation(mask_image)
    return im


class _ShiftedThreshold:
    """Context manager: candidates.adaptive_threshold returns its value plus `delta` HU."""

    def __init__(self, delta: float):
        self.delta = delta
        self.orig = None

    def __enter__(self):
        self.orig = candidates.adaptive_threshold
        orig, delta = self.orig, self.delta

        def shifted(ct, mask):
            thr, stats = orig(ct, mask)
            stats = dict(stats, threshold_hu=round(thr + delta, 1), threshold_shift_hu=delta)
            return thr + delta, stats

        candidates.adaptive_threshold = shifted
        return self

    def __exit__(self, *exc):
        candidates.adaptive_threshold = self.orig
        return False


def run_perturbed(image: sitk.Image, mask_image: sitk.Image, case_id: str, mode: str) -> dict:
    if mode == "mask_erode1":
        return coarse_check.run_stack(image, _morph_mask(mask_image, "erode"), case_id)
    if mode == "mask_dilate1":
        return coarse_check.run_stack(image, _morph_mask(mask_image, "dilate"), case_id)
    if mode == "thr_minus":
        with _ShiftedThreshold(-THRESHOLD_SHIFT_HU):
            return coarse_check.run_stack(image, mask_image, case_id)
    if mode == "thr_plus":
        with _ShiftedThreshold(THRESHOLD_SHIFT_HU):
            return coarse_check.run_stack(image, mask_image, case_id)
    raise ValueError(mode)


def _ref_matches(stack: dict, ref) -> set:
    """Reference ids matched by this stack's prediction at the working cutoff."""
    if ref is None:
        return set()
    pairs = scorer.match(stack["result"]["daughters"], ref["daughters"], config.MATCH_CUTOFF_MM)
    return {ref["daughters"][j]["instance_id"] for _, j, _ in pairs}


def check_case(n: int, modes=PERTURBATIONS, data_dir: str = scorer.DATA_DIR) -> tuple[list, dict]:
    cf = scorer.case_files(n, data_dir)
    cid = cf["case_id"]
    image, mask_image, _ = io_utils.load_case(cf["image"], cf["mask"])
    ref = scorer.load_reference(n)
    base = coarse_check.run_stack(image, mask_image, cid)
    base_refs = _ref_matches(base, ref)
    lines = [f"== {cid}: baseline {len(base['kept'])} kept of {base['inst'].n} patches, threshold {base['cand'].threshold_hu:.0f} HU"
             + (f", {len(base_refs)} of {len(ref['daughters'])} references matched" if ref else "")]
    per_mode = {}
    for mode in modes:
        alt = run_perturbed(image, mask_image, cid, mode)
        head = f"  -- {mode}: {len(alt['kept'])} kept of {alt['inst'].n} patches, threshold {alt['cand'].threshold_hu:.0f} HU"
        body, stats = coarse_check.compare_stacks(base, alt, "baseline", mode)
        if ref:
            alt_refs = _ref_matches(alt, ref)
            stats["ref_lost"] = sorted(base_refs - alt_refs)
            stats["ref_gained"] = sorted(alt_refs - base_refs)
            stats["ref_matched"] = len(alt_refs)
            stats["fp"] = len(alt["kept"]) - len(alt_refs)
            head += f"; references matched {len(alt_refs)}"
            if stats["ref_lost"]:
                head += f", TRUE POSITIVES LOST: {', '.join(stats['ref_lost'])}"
            if stats["ref_gained"]:
                head += f", references newly matched: {', '.join(stats['ref_gained'])}"
        stats["n_kept"] = len(alt["kept"])
        lines.append(head)
        lines += [l for l in body if l.startswith("    LOST") or l.startswith("    GAINED") or "invariants" in l or l.startswith("  matched")]
        per_mode[mode] = stats
    lines.append("")
    return lines, {"modes": per_mode, "n_kept": len(base["kept"]), "ref_matched": len(base_refs) if ref else None, "n_ref": len(ref["daughters"]) if ref else 0}


def report(cases: list, modes=PERTURBATIONS, data_dir: str = scorer.DATA_DIR) -> str:
    lines = [f"Perturbation-stability check, commit {scorer._git_head()}, {time.strftime('%Y-%m-%d %H:%M')}",
             f"Each case is run unperturbed (baseline) and with the mask eroded or dilated by one native voxel, or the threshold shifted by",
             f"-/+{THRESHOLD_SHIFT_HU:g} HU; the perturbed prediction is matched to the baseline at the {config.MATCH_CUTOFF_MM:g} mm cutoff. On the labelled cases the",
             "reference matches are re-counted per perturbation. A measurement of fragility, not a tuning set (SPEC section 3).", ""]
    all_stats = {}
    for n in cases:
        try:
            case_lines, stats = check_case(n, modes, data_dir)
        except Exception as e:  # noqa: BLE001
            case_lines = [f"== subject{n:03d}: FAILED {type(e).__name__}: {e}", "   " + traceback.format_exc().strip().splitlines()[-1], ""]
            stats = None
        print("\n".join(case_lines), file=sys.stderr)
        lines += case_lines
        if stats is not None:
            all_stats[n] = stats
    total_kept = sum(s["n_kept"] for s in all_stats.values())
    lines.append(f"AGGREGATE over {len(all_stats)} cases: {total_kept} baseline branches")
    labelled = {n: s for n, s in all_stats.items() if s["ref_matched"] is not None}
    if labelled:
        lines.append(f"  labelled cases baseline: {sum(s['ref_matched'] for s in labelled.values())} of {sum(s['n_ref'] for s in labelled.values())} references matched, "
                     f"{sum(s['n_kept'] - s['ref_matched'] for s in labelled.values())} false positives")
    for mode in modes:
        ms = {n: s["modes"][mode] for n, s in all_stats.items() if mode in s["modes"]}
        tp = sum(s["tp"] for s in ms.values())
        fn = sum(s["fn"] for s in ms.values())
        fp = sum(s["fp"] for s in ms.values() if "ref_lost" not in s)
        ost = [d for s in ms.values() for d in s["ostium"]]
        ang = [a for s in ms.values() for a in s["direction"]]
        moved = sum(d > 2.0 for d in ost)
        lf = Counter(f for s in ms.values() for f in s["lost_fates"])
        gf = Counter(f for s in ms.values() for f in s["gained_fates"])
        lines.append(f"  {mode}: baseline branches kept {tp} of {total_kept}, lost {fn}, gained {sum(s['fp'] if 'ref_lost' not in s else s['n_kept'] - s['tp'] for s in ms.values())}; "
                     f"ostium moved {np.mean(ost) if ost else float('nan'):.2f} mm mean, {max(ost) if ost else float('nan'):.2f} max, over 2 mm on {moved}; "
                     f"direction moved {np.mean(ang) if ang else float('nan'):.1f} deg mean, over 30 deg on {sum(a > 30 for a in ang)}")
        lines.append(f"      lost because: " + (", ".join(f"{k} {v}" for k, v in lf.most_common()) or "nothing lost")
                     + "; gained from: " + (", ".join(f"{k} {v}" for k, v in gf.most_common()) or "nothing gained"))
        lab = {n: s for n, s in ms.items() if "ref_lost" in s}
        if lab:
            tps = sum(s["ref_matched"] for s in lab.values())
            fps = sum(s["fp"] for s in lab.values())
            lost = [f"{n}/{r}" for n, s in lab.items() for r in s["ref_lost"]]
            gained = [f"{n}/{r}" for n, s in lab.items() for r in s["ref_gained"]]
            lines.append(f"      labelled cases: {tps} references matched, {fps} false positives"
                         + (f"; true positives lost: {', '.join(lost)}" if lost else "; no true positive lost")
                         + (f"; newly matched: {', '.join(gained)}" if gained else ""))
    # branches that flip under any perturbation, by case
    flips = defaultdict(list)
    for n, s in all_stats.items():
        for mode, ms in s["modes"].items():
            if ms["fn"] or ms["fp"]:
                flips[n].append(f"{mode} -{ms['fn']}/+{ms['fp'] if 'ref_lost' not in ms else ms['n_kept'] - ms['tp']}")
    stable = [n for n in all_stats if n not in flips]
    lines.append(f"  cases with no flip under any perturbation: {stable}")
    lines.append("  cases with flips (perturbation -lost/+gained): " + "; ".join(f"{n}: {', '.join(v)}" for n, v in flips.items()))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=[DEFAULT_CASES])
    ap.add_argument("--modes", nargs="*", default=list(PERTURBATIONS))
    ap.add_argument("--data-dir", default=scorer.DATA_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    text = report(scorer._parse_cases(args.cases), tuple(args.modes), args.data_dir)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
