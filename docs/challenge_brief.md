# Branchseed Challenge — Discover Aortic Branch Origins

(Text transcription of the organisers' `Branchseed challenge.pdf`, kept
alongside the implementation spec and design doc as the primary source of
truth for scoring and requirements.)

## The challenge in one sentence

Given a CT volume and a binary mask containing only the parent abdominal
aorta, detect every eligible artery that directly leaves the aorta and return
each one as a separate daughter instance.

## Background

The abdominal aorta gives rise to several arteries. Their number, position and
direction vary between patients, and a scan may show only part of the aorta.
This complicates surgeries to repair them as the anatomy can become
unpredictable.

For example, one of these is the superior mesenteric artery (SMA), which
branches from the front of the aorta. CT imaging can be thought of as taking
many thin cross-sectional slices through the body. Before a CT angiogram
(CTA), an iodine-based contrast agent is injected into the bloodstream. When
the scan is taken at the right time, blood inside the arteries appears bright
on CT. Instead of seeing the whole artery as a long tube, an axial CT slice
shows the vessels where that particular slice intersects them.

For a computer vision system, the task is essentially: **identify these
bright vascular structures, determine which vessel is which, and follow them
across consecutive CT slices**. This challenge does not ask you to search for
a fixed list of named arteries. Your system must discover the branch origins
that are actually present in each case.

## What you will receive

Approximately 25 paired NIfTI volumes:

```
data/
  subject001/
    orig1.nii
    mask1.nii
  subject002/
    orig2.nii
    mask2.nii
  ...
```

* `orig1.nii` is the CT volume.
* `mask1.nii` is a binary mask of the parent aortic lumen only: 1 = aorta,
  0 = everything else.
* The image and mask have the same grid and physical-coordinate system.
* The daughter arteries are visible in the CT image but are not included or
  labelled in the supplied mask.
* Anatomical coverage varies. Some cases contain a shorter or longer section
  of the aorta than others.
* A small development subset will include example reference outputs. Final
  evaluation cases and their reference annotations will remain hidden.

The aorta mask is a search anchor. It does not contain the daughter vessels,
so the mask alone is not sufficient to solve the task.

## Your task

For each previously unseen case, your system must automatically:

1. examine the CT around the supplied aorta;
2. detect every eligible artery arising directly from the aorta;
3. assign each detected artery a unique daughter-instance ID;
4. link every daughter instance to the parent aorta;
5. return the centre of its origin, a point showing its initial direction,
   and an estimate of the local daughter-vessel radius.

Do not assign anatomical names. A detected artery should be reported as
`branch_001`, `branch_002`, and so on.

## Definitions

* **Parent aorta:** the lumen represented by the supplied binary mask.
* **Direct daughter:** an artery whose lumen connects directly to the
  supplied parent aorta.
* **Ostium centre:** the centre of the opening where a direct daughter leaves
  the parent aorta.
* **Daughter seed:** a point at the centre of the daughter lumen, 5 mm
  outward from the ostium along the daughter path.
* **Daughter radius:** an estimate of the local lumen radius at the daughter
  seed, reported in millimetres.
* **Daughter instance:** one independently detected branch represented by one
  unique ID.

A branch is eligible for scoring when its contrast-filled lumen can be
followed for at least 5 mm beyond the aortic wall and its origin meets the
minimum size specified with the final dataset. These rules will be applied
consistently to the reference annotations.

For each eligible daughter, trace the proximal branch for up to 10 mm beyond
the ostium or until the first downstream bifurcation, whichever occurs first.
This proximal path should be used to estimate the daughter seed, local radius
and initial direction.

## Required output

Produce one file per case, this information could be in JSON (like below), or
something creative!

```json
{
  "case_id": "subject001",
  "parent": {
    "instance_id": "aorta"
  },
  "daughters": [
    {
      "instance_id": "branch_001",
      "parent_instance_id": "aorta",
      "ostium_xyz_mm": [12.4, -31.8, 184.6],
      "seed_xyz_mm": [15.1, -29.7, 181.2],
      "radius_mm": 2.7,
      "direction_xyz": [0.56, 0.43, -0.71]
    }
  ]
}
```

Requirements:

* `ostium_xyz_mm` and `seed_xyz_mm` must be reported in physical millimetres
  using the physical coordinate system returned by SimpleITK. Use
  `SimpleITK.TransformIndexToPhysicalPoint` when converting voxel locations
  to physical coordinates. Do not report voxel indices.
* `radius_mm` must represent the estimated local lumen radius at the
  daughter seed and must be reported in millimetres.
* `direction_xyz` must be a unit vector pointing from the ostium into the
  daughter vessel.
* Each real daughter must appear once only.
* Each prediction must have a unique instance ID and
  `parent_instance_id = "aorta"`.
* If no eligible daughter is visible, return an empty `daughters` list.

Teams must also provide a simple visual check for at least three cases
showing the aorta mask, detected ostia and daughter-direction arrows. The
visualisation is for verification; an elaborate clinical interface is not
required.

## Important cases

* A short or long supplied aortic segment may contain a different number of
  daughters.
* The flat superior and inferior ends created by cropping are not branch
  origins.
* Two nearby origins must be returned as two instances when they are
  separate at the aortic wall.
* A common trunk has one direct aortic origin, even if it divides shortly
  afterwards.
* A vessel arising from another daughter is not a direct aortic daughter.
* A vessel that is absent from the image or outside the supplied coverage
  must not be guessed.
* The terminal division of the aorta into the iliac arteries is outside the
  core task and may be evaluated separately as an optional extension.

## Minimum working prototype

Your submission must:

* accept a new CT and parent-aorta mask without manual point placement;
* produce valid JSON (or any format you find fit, be creative!);
* display information in a unique way that will be useful for clinicians;
* detect a variable number of daughter instances;
* preserve the input physical-coordinate system;
* run on the complete evaluation set without case-specific edits;
* run on a standard laptop without a GPU.

You may use classical image processing, vessel-enhancement filters, graph or
search algorithms, lightweight machine learning, or a hybrid approach. No
particular method is required.

## Evaluation

Reference annotations will contain one ostium centre, a short proximal
centreline and a local radius measurement at the daughter seed for every
eligible direct daughter. Predictions will be matched to references
one-to-one, so duplicate detections count as false positives.

| Category | Weight | What is assessed |
|---|---|---|
| Branch discovery | 45% | Precision, recall and F1 for detecting the correct number of daughter instances |
| Ostium localisation | 25% | Physical distance between predicted and reference ostium centres |
| Daughter-instance quality | 15% | Whether the seed lies on the matched daughter, the predicted direction follows its proximal path and the estimated local radius is consistent with the daughter lumen |
| Compute efficiency | 10% | Runtime and peak memory on the organiser's CPU-only system |
| Reproducibility | 5% | Valid output, documented setup and successful execution on unseen cases |

The organisers will use a hidden test set and a standard environment with
four CPU cores, 8 GB RAM, no GPU and no internet access. The initial runtime
target is an average of no more than 60 seconds per case; the final limit
will be confirmed after the organiser baseline is tested.

## Submission

Submit:

* source code;
* dependency or environment file;
* a short README with one setup command and one run command;
* predictions for the development set;
* the required visual checks;
* a five-minute demonstration explaining your method, its runtime and known
  failure cases.

Your program must support:

```
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

## Out of scope

* segmenting the parent aorta;
* assigning anatomical vessel names;
* reconstructing the complete distal vascular tree;
* predicting branches that are not visible.

The goal is deliberately narrow:

*Find every eligible artery that directly leaves the supplied aorta, keep the
daughters separate, and represent each origin accurately enough to be used as
a machine-readable branch instance.*
