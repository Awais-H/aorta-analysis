#!/usr/bin/env python3
"""Write the synthetic suite out as a dataset directory plus reference JSON.

Gives a runnable, self-contained development set with exact ground truth, in
the same layout as the supplied data, so ``predict_all.py`` and
``evaluate.py`` can be exercised end to end before any annotated case exists.
The volumes are regenerated rather than committed; the generator is
deterministic, so the same command reproduces the same bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

import phantoms  # noqa: E402

# name -> (spec builder, one-line description of what it tests)
SUITE = {
    "phantom01_standard": (phantoms.standard,
                           "four branches at four levels, distractors, two cut faces"),
    "phantom02_arched": (phantoms.arched,
                         "curved parent: the frame must not degenerate"),
    "phantom03_nearby_pair": (phantoms.nearby_pair,
                              "two origins 8 mm apart stay two instances"),
    "phantom04_common_trunk": (phantoms.common_trunk,
                               "a trunk dividing at 3 mm is one origin"),
    "phantom05_daughter_of_daughter": (phantoms.daughter_of_daughter,
                                       "a vessel off a daughter is not a direct daughter"),
    "phantom06_short_stub": (phantoms.short_stub,
                             "a bump dying at 3 mm fails the 5 mm rule"),
    "phantom07_leaking": (phantoms.leaking,
                          "a stub opening into a blob is discarded, not truncated"),
    "phantom08_near_cut_face": (phantoms.near_cut_face,
                                "a real origin 14 mm from a cut face survives exclusion"),
    "phantom09_no_branches": (phantoms.no_branches,
                              "a bare tube yields a valid empty list"),
}


def reference_payload(case_id: str, truth) -> dict:
    daughters = []
    for n, item in enumerate((t for t in truth if t["eligible"]), start=1):
        ostium = np.asarray(item["ostium_xyz_mm"], dtype=float)
        direction = np.asarray(item["direction_xyz"], dtype=float)
        daughters.append({
            "instance_id": f"branch_{n:03d}",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [round(float(v), 4) for v in ostium],
            "seed_xyz_mm": [round(float(v), 4) for v in item["seed_xyz_mm"]],
            "radius_mm": round(float(item["radius_mm"]), 4),
            "direction_xyz": [round(float(v), 6) for v in direction],
            "proximal_centreline": [
                [round(float(v), 4) for v in ostium + step * direction]
                for step in np.linspace(0.0, 10.0, 11)
            ],
        })
    return {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": daughters}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic development set.")
    parser.add_argument("--data", default=str(ROOT / "data" / "phantoms"))
    parser.add_argument("--references", default=str(ROOT / "references" / "phantoms"))
    args = parser.parse_args(argv)

    data, references = Path(args.data), Path(args.references)
    references.mkdir(parents=True, exist_ok=True)

    index = {}
    for case_id, (builder, description) in SUITE.items():
        case_dir = data / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        phantom = phantoms.build_phantom(builder())
        sitk.WriteImage(phantom.image, str(case_dir / "orig.nii.gz"))
        sitk.WriteImage(phantom.mask, str(case_dir / "mask.nii.gz"))
        payload = reference_payload(case_id, phantom.truth)
        (references / f"{case_id}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        index[case_id] = {"tests": description, "eligible_daughters": len(payload["daughters"])}
        print(f"{case_id}: {len(payload['daughters'])} eligible daughters - {description}")

    (references / "_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
