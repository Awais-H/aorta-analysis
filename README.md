# Branchseed challenge

Finds every artery that leaves the supplied abdominal aorta and reports each origin as a machine-readable branch instance (ostium, seed 5 mm in, direction, radius). Design and reasoning: [SPEC.md](SPEC.md). Challenge text: [docs/Branchseed_challenge.pdf](docs/Branchseed_challenge.pdf).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or newer, CPU only, no torch.

## Run

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Optional flags: `--report-dir DIR` writes the verification PNG and the clinician HTML report; `--meta-output FILE` writes per-stage timings, the chosen threshold, the mask fragment log and the rejection log; `--case-id` overrides the id inferred from the `subjectNNN` folder name. Input files may be gzip-compressed regardless of extension. run.py never crashes on a case: on any error it logs the traceback, writes an empty daughters list and exits 0.

## Test

```bash
python -m pytest tests
```

One fake-data test file per module, built on a synthetic phantom (bright cylinder aorta with one side branch) in `tests/conftest.py`.

## Layout

| File | Stage |
|---|---|
| `run.py` | CLI entry, writes JSON, never crashes |
| `pipeline.py` | orchestrates stages, per-stage timing |
| `config.py` | every numeric constant with its justification (the only place thresholds live) |
| `io_utils.py` | NIfTI reading (gzip sniffing, non-orthonormal header fallback), crop, resample, the only index-to-mm conversion |
| `candidates.py` | D1 adaptive threshold, D2 shell and ring-removing opening |
| `instances.py` | D3 wall-patch components and watershed growth |
| `ostium.py` | D4 ostium localisation (stub) |
| `tracing.py` | D5 seed, direction, radius (stub) |
| `filters.py` | D6 false-positive rules (stub: eligibility only) |
| `frame.py` | centreline, clock frame, mm to (height, clock) (stub: slice centroids) |
| `report.py` | verification PNG and HTML report (stub: table only) |
| `scorer.py` | D7 Hungarian matching, metrics, invariants |
| `triage.py` | per-case atlas tool used to build `docs/atlas_all.csv` |
