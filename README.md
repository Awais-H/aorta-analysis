# Branchseed challenge

Finds every artery that leaves the supplied abdominal aorta and reports each origin as a machine-readable branch instance (ostium, seed 5 mm in, direction, radius). Current state and what remains: [docs/HANDOFF.md](docs/HANDOFF.md). Design and reasoning: [SPEC.md](SPEC.md). Challenge text: [docs/Branchseed_challenge.pdf](docs/Branchseed_challenge.pdf).

On the five labelled dev cases (19 to 23) at a 5 mm ostium cutoff: 17 of 19 reference branches found, 13 false positives (a looping vessel the annotators also could not place, borderline 2 mm origins, and small vessels the draft references do not cover), mean ostium error 1.35 mm, under 10 s per case on a laptop CPU. Details in `results/`: scores, ledger, sweeps, invariants, the predictions for all 25 dev cases (`results/predictions/`) and the visual checks (`results/visual_checks/`).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or newer (developed on 3.13), CPU only, no torch. An offline install with vendored wheels is planned once the scoring machine's platform is known.

## Run

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Optional flags: `--report-dir DIR` writes the verification PNG (mask, ostia, direction arrows, centreline), the clock map PNG and a self-contained HTML report (branch table with clock position, height, radius and flags; spacing along the aorta; interactive 3D view, works offline); `--meta-output FILE` writes per-stage timings, the chosen threshold, the mask fragment log and the rejection log; `--case-id` overrides the id inferred from the `subjectNNN` folder name. Input files may be gzip-compressed regardless of extension. run.py never crashes on a case: on any error it logs the traceback, writes an empty daughters list and exits 0.

## Test

```bash
python -m pytest tests
```

One fake-data test file per module, built on a synthetic phantom (bright cylinder aorta with one side branch) in `tests/conftest.py`.

## Score and check

```bash
python scorer.py predict --cases 19-23
```

```bash
python scorer.py score --cases 19-23 --out results/dev_scores.txt
```

```bash
python scorer.py invariants --cases all --out results/invariants.txt
```

```bash
python scorer.py ledger --cases 19-23 --out results/reference_ledger.txt
```

```bash
python sweeps.py --out results/sweeps.txt
```

`predict` runs run.py per case into `out/predictions/`. `score` matches predictions one-to-one against the draft references in `docs/references` (Hungarian on ostium distance at 3, 5 and 8 mm) and reports precision, recall, F1, ostium distance, direction error, seed-on-branch (via the per-case daughter label volume) and radius error where the reference has one. `invariants` runs the SPEC.md D7 checks on any case, referenced or not. `ledger` writes, for every reference branch, the nearest wall patch, its fate in the filters and every measurement the rules saw, plus the surviving false positives. `sweeps.py` varies one constant at a time and reports the TP/FP curve (SPEC.md section 3: the shape matters, not the peak). All four result files are committed after every change to the pipeline.

## Clinician report

```bash
python run.py --image data/subject022/orig22.nii --aorta-mask data/subject022/mask22.nii --output out/report/subject022.json --report-dir out/report
```

Writes `subject022_check.png` (coronal and sagittal projections: candidate voxels, aorta mask, centreline, ostia, 10 mm direction arrows), `subject022_clock.png` (the aortic wall unrolled: clock position across, height from the superior cut down, markers sized by radius, green = passed every rule, amber = flagged, brackets with the spacing between branches, aortic diameter alongside) and `subject022_report.html` (branch table in clinician terms with the flags explained, both images, an interactive 3D view of the mask with ostia and directions; one file, opens without a network). Clock convention: 12 anterior, 3 patient's left, 6 posterior, 9 patient's right, measured perpendicular to the local centreline. Branch IDs carry no anatomical meaning.

## Review a case by eye

```bash
python gallery.py --cases 22 --pred-dir out/predictions --out out/gallery
```

Writes sheets of CT crops, six candidates per sheet: axial through the ostium, axial through the seed 5 mm out, coronal and sagittal, with the mask outlined and the ostium, seed and direction drawn. Add `--rejected` to include every rejected wall patch. `review_markers.py` writes the same candidates as 2 mm spheres on the native grid (kept = labels 1..N, rejected = 101..) with an ITK-SNAP label file, plus a per-branch table with voxel indices for the slice sliders.

To view in ITK-SNAP (the sponsor's recipe): File → Open Main Image → the case's `origNN.nii.gz`; Segmentation → Open Segmentation → `subjectNNN_pred_markers.nii.gz`; Segmentation → Label Definitions → Import → `subjectNNN_labels.txt`. In Cursor Inspector type the voxel index from the review table; in Zoom Inspector set 6 px/mm and Center on cursor. For the 3D view, Tools → Preferences → 3D Rendering, turn Gaussian smoothing off, then Update. Workspace → Save Workspace keeps the setup.

## Layout

| File | Stage |
|---|---|
| `run.py` | CLI entry, writes JSON, never crashes |
| `pipeline.py` | orchestrates stages, per-stage timing |
| `config.py` | every numeric constant with its justification (the only place thresholds live) |
| `io_utils.py` | NIfTI reading (gzip sniffing, non-orthonormal header fallback), crop, resample, the only index-to-mm conversion |
| `candidates.py` | D1 adaptive threshold, D2 shell and ring-removing opening |
| `instances.py` | D3 wall-patch components and watershed growth |
| `ostium.py` | D4 ostium localisation |
| `tracing.py` | D5 seed, direction, radius |
| `filters.py` | D6 false-positive rules 1 to 7 |
| `frame.py` | geodesic centreline, end faces, clock frame (12 = patient anterior at each level), aortic radius profile, spacing |
| `report.py` | verification PNG, clock map, self-contained HTML report with Plotly 3D |
| `scorer.py` | D7 Hungarian matching, metrics, invariants, reference ledger |
| `sweeps.py` | section 3 sensitivity sweeps |
| `gallery.py` | D7 failure gallery: crop sheets per candidate |
| `review_markers.py` | ITK-SNAP marker volumes and per-branch review tables |
| `triage.py` | per-case atlas tool used to build `docs/atlas_all.csv` |
