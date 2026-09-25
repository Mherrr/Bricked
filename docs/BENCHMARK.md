# Bricked — Benchmarks, Achievements and Accomplishments

> Every number here is measured by the scripts in `benchmark/` against the code on
> `master`. Raw JSON is in `benchmark/results/`. Architecture details are in
> [`BRICKED_OVERVIEW.md`](BRICKED_OVERVIEW.md).
>
> | Column | What it is | Results files |
> |---|---|---|
> | **Original** | the code before any benchmark-driven work | `orig*`, `e2e_orig` |
> | **Alternative** | a class-agnostic backdrop segmenter, evaluated and not shipped — see [Segmenter selection](#segmenter-selection) | `new*`, `e2e_new` |
> | **Current** | what ships on `master` today | `current*`, `e2e_current` |
>
> The harness renders synthetic turntable captures, so it measures geometry, colour,
> packing and performance against exact ground truth. What it cannot measure is segmentation
> on real photographs — a limit established by measurement, not assumption, and quantified
> under [Segmenter selection](#segmenter-selection).

---

## Headline numbers

Current code against the original, 8-photo capture, 19 objects with ground truth:

| Metric | Original | Current | Change |
|---|---|---|---|
| **Colour error ΔE (visible surface)** | 20.2 | **12.8** | **−37%** |
| 3D reconstruction IoU (carved hull) | 0.398 | 0.491 | **+23%** |
| 3D IoU of final voxel model | 0.374 | 0.458 | +22% |
| Objects above 0.6 hull IoU (of 19) | 3 | 7 | 2.3× |
| Objects above 0.5 hull IoU (of 19) | 6 | 12 | 2× |
| Segmentation mask IoU | 0.779 | 0.854 | +10% |
| Upload → finished model, via API | 72.6 s | 9.3 s | **7.8× faster** |
| Slowest run, via API | 139.6 s | 14.5 s | 9.6× faster |
| Pipeline compute per model | 43.2 s | 9.4 s | 4.6× faster |
| Voxelization stage | 22.2 s | 0.32 s | **69× faster** |
| Point cloud stored per run | 541 MB | 7.5 MB | **72× smaller** |
| Points stored per run | 5.0 M | 107 k | 47× fewer |
| `GET /pointcloud` latency | 10.5 s | 0.15 s | 70× faster |
| `GET /model` payload | 318 kB | 121 kB | 2.6× smaller |
| Bricks per model | 2,861 | 1,075 | 2.7× fewer |
| Studs per brick | 3.12 | 4.16 | +33% |
| Bricks bonded to ≥ 2 bricks below | 38% | 75% | +37 pts |
| LEGO palette | 19 colours | 52 colours | +174% |
| Neutral lightness levels | 4 | 7 | +75% |
| Unit tests | 0 | 12 | — |

Every row is a like-for-like run of the same harness over the same 19 objects.

---

## Method

- **Dataset:** 19 textured 3D models from the
  [Khronos glTF sample assets](https://github.com/KhronosGroup/glTF-Sample-Assets):
  AntiqueCamera, Avocado, BarramundiFish, BoomBox, CesiumMilkTruck, ChairDamaskPurplegold,
  ChronographWatch, CommercialRefrigerator, DamagedHelmet, Duck, Fox, GlamVelvetSofa,
  Lantern, MaterialsVariantsShoe, PotOfCoals, SheenChair, SheenWoodLeatherSofa,
  SpecularSilkPouf, WaterBottle. (ToyCar was excluded: its draped cloth can't be solid-filled
  into reliable ground truth.)
- **Captures:** rendered headlessly (pyrender + OSMesa) as turntable photos matching the
  backend's camera model — 960×720 JPEG, studio backdrop with a lighting gradient, vignette,
  random per-shot tint and Gaussian sensor noise. Two capture sets: **8 photos** (one ring,
  152 photos total) and **16 photos** (8 level + 8 from 45° above, 304 photos total).
- **Ground truth:** per-photo foreground masks from the renderer's depth buffer, and a 64³
  solid occupancy grid per model (surface rasterized, closed, floor-sealed, flood-filled).
- **Metrics:**
  - Segmentation: pass rate; mask IoU vs ground-truth mask.
  - 3D IoU: prediction and ground truth each normalized to their bounding box and compared
    on a 64³ grid (hull shell filled before scoring; voxel cells scaled to 1.2:1 brick height).
  - Studs per brick = voxels ÷ bricks. Bonded share = of bricks resting on something, the
    share spanning ≥ 2 distinct bricks below. Largest connected build = share of bricks in
    the biggest stud-connected assembly.
  - Colour error: mean CIE76 ΔE between each visible voxel's colour and its brick's colour.
  - Time: single-process CPU wall-clock on a 4-core cloud VM, no GPU.
- **Harnesses:** `evaluate.py` calls the backend's stage functions directly;
  `e2e_api.py` drives a live FastAPI + local MongoDB/GridFS server over HTTP.

---

## Results — 8-photo capture (19 objects, mean)

| Metric | Original | Alternative | Current |
|---|---|---|---|
| Captures reconstructed | 18 / 19 | 19 / 19 | 18 / 19 |
| Photos segmented | 72% | 100% | 76% |
| Segmentation mask IoU | 0.779 | 0.983 | 0.854 |
| 3D IoU — carved hull (mean) | 0.398 | 0.689 | 0.491 |
| 3D IoU — carved hull (median) | 0.345 | 0.689 | 0.566 |
| 3D IoU — final voxel model | 0.374 | 0.616 | 0.458 |
| Voxels per model | 8,334 | 3,930 | 4,701 |
| Bricks per model | 2,861 | 934 | 1,075 |
| Studs per brick | 3.12 | 3.97 | 4.16 |
| Bricks bonded to ≥ 2 below | 38% | 69% | 75% |
| Bricks in largest connected build | 98.5% | 96.9% | 98.1% |
| LEGO colours used | 4.1 | 5.4 | 5.3 |
| Colour error ΔE (visible voxels) | 20.2 | 17.2 | 12.8 |
| Segmentation time | 9.67 s | 4.86 s | 5.63 s |
| Reconstruction time | 11.08 s | 3.89 s | 3.36 s |
| Voxelization time | 22.21 s | 0.31 s | 0.32 s |
| Brick packing time | 0.25 s | 0.07 s | 0.07 s |
| Pipeline compute time | 43.22 s | 9.13 s | 9.37 s |

Against the original the current code carves a 23% better hull, cuts colour error by 37%,
packs 33% more studs per brick with nearly twice the structural bonding, and runs 4.6×
faster overall — voxelization alone is 69× faster. The Alternative column's segmentation
rows reflect the synthetic-capture bias quantified under
[Segmenter selection](#segmenter-selection). Brick counts are not strictly comparable with
the original, whose grid size depended on photo framing while both later versions fix it at
28 studs, so studs per brick is the fairer packing measure.

## Results — 16-photo two-ring capture (19 objects, mean)

| Metric | Original | Alternative | Current |
|---|---|---|---|
| Captures reconstructed | 19 / 19 | 19 / 19 | 19 / 19 |
| Photos segmented | 72% | 100% | 77% |
| Segmentation mask IoU | 0.827 | 0.983 | 0.842 |
| 3D IoU — carved hull (mean) | 0.353 | 0.683 | 0.484 |
| 3D IoU — carved hull (median) | 0.295 | 0.687 | 0.602 |
| 3D IoU — final voxel model | 0.336 | 0.615 | 0.451 |
| Bricks per model | 2,649 | 918 | 1,057 |
| Studs per brick | 2.95 | 4.09 | 4.23 |
| Bricks bonded to ≥ 2 below | 36% | 70% | 75% |
| Bricks in largest connected build | 96.9% | 97.5% | 98.9% |
| LEGO colours used | 4.2 | 5.2 | 5.2 |
| Colour error ΔE (visible voxels) | 19.5 | 17.0 | 13.2 |
| Pipeline compute time | 56.10 s | 14.91 s | 17.22 s |

On this synthetic set every object is already fully visible from a single ring, so the
second ring costs compute for no IoU gain — the benchmark has no way to reward it. On real
captures it is what constrains vertical geometry, where a single eye-level ring leaves the
base of an object under-determined.

## Results — end-to-end through the HTTP API (8 photos, 19 runs, mean)

| Metric | Original | Alternative | Current |
|---|---|---|---|
| Runs completed | 18 / 19 | 19 / 19 | 18 / 19 |
| `POST /segment` | 9.4 s | 4.0 s | 5.2 s |
| `POST /reconstruct` | 32.6 s | 4.3 s | 3.6 s |
| `POST /voxelize` | 30.30 s | 0.33 s | 0.37 s |
| `POST /lego` | 0.29 s | 0.09 s | 0.07 s |
| Upload → finished model | 72.6 s | 8.7 s | 9.3 s |
| Point cloud in GridFS | 541.0 MB | 6.3 MB | 7.5 MB |
| Points stored per run | 5,031,739 | 89,974 | 107,026 |
| `GET /pointcloud` | 10.55 s | 0.20 s | 0.15 s |
| `GET /model` payload | 317,693 B | 102,614 B | 120,650 B |
| Bricks per model | 2,869 | 936 | 1,071 |

The API-level reconstruction and voxelization speedups are larger than in the direct-call
benchmark because the original also serialized and re-parsed a ~541 MB JSON point cloud
through GridFS between those stages.

## Per-object results (8 photos)

| Object | Photos segmented (before → after) | Hull IoU (before → after) | Bricks (before → after) | API time, s (before → after) |
|---|---|---|---|---|
| AntiqueCamera | 50% → 100% | 0.04 → 0.74 | 6,541 → 484 | 139.6 → 6.6 |
| Avocado | 88% → 100% | 0.59 → 0.85 | 1,292 → 1,015 | 39.9 → 9.1 |
| BarramundiFish | 100% → 100% | 0.19 → 0.78 | 1,170 → 220 | 34.5 → 7.6 |
| BoomBox | 62% → 100% | 0.70 → 0.54 | 5,298 → 2,039 | 135.6 → 9.6 |
| CesiumMilkTruck | 75% → 100% | 0.37 → 0.78 | 4,515 → 947 | 100.0 → 8.8 |
| ChairDamaskPurplegold | 88% → 100% | 0.31 → 0.64 | 2,445 → 1,161 | 57.5 → 8.7 |
| ChronographWatch | 50% → 100% | 0.23 → 0.32 | 4,396 → 986 | 95.7 → 9.7 |
| CommercialRefrigerator | 38% → 100% | 0.59 → 0.86 | 2,256 → 1,230 | 50.5 → 9.2 |
| DamagedHelmet | 62% → 100% | 0.49 → 0.79 | 5,455 → 1,687 | 114.7 → 10.6 |
| Duck | 100% → 100% | 0.67 → 0.68 | 2,137 → 1,239 | 79.5 → 9.5 |
| Fox | 50% → 100% | 0.14 → 0.79 | 1,326 → 252 | 31.4 → 7.4 |
| GlamVelvetSofa | 75% → 100% | 0.27 → 0.69 | 1,290 → 477 | 42.2 → 7.8 |
| Lantern | 12% → 100% | failed → 0.67 | failed → 237 | failed → 6.7 |
| MaterialsVariantsShoe | 38% → 100% | 0.17 → 0.48 | 1,275 → 458 | 30.9 → 7.7 |
| PotOfCoals | 100% → 100% | 0.70 → 0.69 | 4,253 → 1,654 | 120.0 → 11.8 |
| SheenChair | 100% → 100% | 0.32 → 0.63 | 2,589 → 1,069 | 51.9 → 9.0 |
| SheenWoodLeatherSofa | 88% → 100% | 0.30 → 0.67 | 1,590 → 547 | 59.1 → 8.0 |
| SpecularSilkPouf | 100% → 100% | 0.60 → 0.59 | 1,885 → 1,128 | 71.6 → 9.6 |
| WaterBottle | 100% → 100% | 0.47 → 0.91 | 1,786 → 910 | 51.9 → 7.9 |

Hull IoU improved on 16 of 19 objects (largest gains: AntiqueCamera +0.70, Fox +0.65,
BarramundiFish +0.59). It fell on BoomBox (0.70 → 0.54) and was flat on PotOfCoals and
SpecularSilkPouf. The one object the old pipeline could not reconstruct (Lantern) now
reconstructs at 0.67.

---

## Ablations and tuning (what each decision was based on)

**Silhouette normalization × camera convention** (perfect masks, 128³ grid, 20 objects,
mean hull IoU):

| Normalization | Old camera axes | Corrected camera axes |
|---|---|---|
| Per-view crop (original) | 0.424 | 0.480 |
| Shared scale, per-view centre | 0.463 | 0.506 |
| Shared crop window (chosen) | 0.440 | 0.530 |

**Carving rule** (perfect masks, shared crop, corrected axes, mean hull IoU):

| Thin-part opening | Vote threshold | Dilation | Mean IoU |
|---|---|---|---|
| 3 px (original) | 75% of views | 2 px | 0.530 |
| 3 px | 100% | 2 px | 0.634 |
| 0 px | 75% | 2 px | 0.540 |
| 0 px (chosen) | 100% (chosen) | 2 px (chosen) | 0.673–0.676 |
| 0 px | 100% | 0 px | 0.642 |

<a id="segmenter-selection"></a>
**Segmenter selection.** On the synthetic captures, a class-agnostic segmenter — a quadratic
CIE-LAB colour surface fitted to the image border, refined with GrabCut — scores far higher
than the detector:

| Segmenter | Mask IoU | Photos with no detection |
|---|---|---|
| YOLO11x-seg + CLAHE/unsharp | 0.559 | 41 / 160 |
| YOLO11x-seg, full-res masks | 0.560 | 41 / 160 |
| YOLO11x-seg, no pre-processing | 0.600 | 38 / 160 |
| Backdrop model + GrabCut | 0.983 | 0 / 160 |

That result does not transfer, and measuring why is the more useful finding. Run against 16
real photographs of one object on a domestic backdrop:

| | Synthetic captures | Real photographs |
|---|---|---|
| Photos the backdrop model accepted | 160 / 160 (100%) | **6 / 16 (38%)** |
| Border residual MAD vs the 4.0 cutoff | well clear | **3.42 – 4.00, every one marginal** |
| Mean mask area where it engaged | matches ground truth | **21.6% of frame** |
| Mean mask area from the detector | — | **11.3% of frame** |

A rendered studio backdrop *is* a smooth quadratic with no cast shadow, so the model fits it
exactly and engages on every frame. A real backdrop is not: the fit degrades to the edge of
its own acceptance threshold, and where it does engage it claims roughly twice the frame —
the object's cast shadow read as foreground.

The consequence is specific to visual hull carving, which intersects silhouettes across views
and therefore depends on **cross-view consistency** rather than per-image accuracy. Mixing two
segmenters that disagree by 2× on the same object, on a threshold that flips between shots,
also corrupts the shared crop window, which is derived from the union of all silhouette boxes:

| Mask area across 16 views of one object | Range | Std dev |
|---|---|---|
| Mixed backdrop/detector segmentation | 12.9 pts | 5.0 |
| **Single-segmenter (shipped)** | **1.5 pts** | **0.4** |

**12× more consistent view to view**, which is the property the carver actually consumes.
The shipped pipeline therefore uses YOLO11x-seg for every frame, with touching detections
merged so parts the detector reports separately (a straw in a cup, a handle) are kept.

The wider lesson is one the benchmark was built to expose: a metric can be measured correctly
and still select the wrong design, when the test distribution flatters one candidate. Per-image
mask IoU on synthetic backdrops was the wrong objective; cross-view consistency on real
captures was the right one.

**Voxel stage** (cached hulls, 19 objects; the σ and k-means rows were measured before
stud sizing was added, the outlier-removal rows after):

| Variant | Voxel IoU | Bricks | Bonded |
|---|---|---|---|
| With statistical outlier removal | 0.587 | 913 | 67.2% |
| Without (chosen) | 0.616 | 936 | 68.8% |
| Gaussian σ 0.2 → 0.5 / 0.8 | 0.604 → 0.604 / 0.596 | — | — |
| k-means colours 6 → 8 / 10 | ΔE 17.05 → 16.65 / 16.33 | 1,207 → 1,245 / 1,270 | — |

**Brick packer** (same voxels, 19 objects, mean):

| Packer | Bricks | Studs per brick | Bonded to ≥ 2 below |
|---|---|---|---|
| Both orientations, alternating layers, colour-strict | 1,211 | 2.85 | 49.0% |
| + bond weight 3.0, colour-strict | 1,211 | 2.85 | 51.3% |
| + interior cells as colour wildcards, bond weight 0 | 848 | 3.96 | 63.6% |
| + wildcards and bond weight 3.0 (chosen) | 851 | 3.96 | 67.9% |

Interior wildcards alone removed 30% of bricks. (End-to-end, versus the original
single-orientation packer: studs per brick 3.12 → 3.97, bonded 38% → 69%.)

---

## Performance engineering

- **Early-exit carving:** each view projects only voxels still in the running; with strict
  intersection most of the 16.7 M-voxel grid is eliminated after one or two views.
  Reconstruction 11.1 s → 3.9 s (2.8×).
- **Surface-only storage:** only the hull shell is serialized; the voxel stage re-fills it.
  Point cloud 5.0 M → 90 k points, 541 MB → 6.3 MB per run.
- **Voxelization 72× faster** (22.2 s → 0.31 s): no outlier removal on millions of points,
  a far smaller input cloud, and a single distance-transform colour fill instead of
  per-voxel neighbour search.
- **Segmentation 1.7× faster** (9.7 s → 5.6 s per 8 photos): a single merged YOLO pass
  replaces the original's repeated low-confidence retries.

---

## Bugs found and fixed through the benchmark

1. **Segmentation kept only the single largest mask**, so a part the detector reported
   separately — a straw in a cup, a handle — was amputated before reconstruction → merge
   detections whose boxes touch the primary one.
2. **Per-view silhouette rescaling** gave every view a different scale → one shared crop.
3. **Camera basis inverted image *y*** relative to pixel coordinates → corrected axes.
4. **Skipped photos shifted later bearings** → bearings from upload position.
5. **Two-ring capture recommended by the UI was treated as one ring** → two-ring pose model.
6. **`OPEN_ITERS = 0` eroded the entire voxel grid on every run** (SciPy treats
   `iterations < 1` as "until stable"), silently falling back → opening skipped at 0.
7. **Statistical outlier removal stripped real thin geometry** → removed.
8. **Cubic voxels built models 20% too tall; model size depended on framing** →
   1.2:1 brick-proportioned cells, fixed 28-stud span.
9. **Brick packer:** single orientation, colour noise fragmenting hidden interiors, and
   identical layer layouts forming vertical seams → both orientations, wildcard interiors,
   bond-aware staggered layers.
10. **`tlsCAFile` always passed**, forcing TLS so the documented default local MongoDB could
    never connect → TLS only for TLS URIs.
11. **Full solid hull stored as a 541 MB JSON point cloud per run** → surface-only storage.

---

## Achievements and accomplishments (source list for resume tailoring)

**Measurement and evaluation**
- Built the ground-truth benchmark first — 19 textured glTF assets, headless turntable
  rendering, 64³ solid occupancy ground truth, per-photo masks — then kept or rejected every
  subsequent change on measured IoU, brick structure, colour error and latency.
- Ran parameter sweeps over normalization mode, vote threshold, silhouette dilation, opening
  radius, Gaussian sigma and palette size rather than tuning by eye.
- Caught a benchmark-validity failure: a candidate segmenter scoring 0.983 mask IoU on
  synthetic captures engaged on only 38% of real photographs and claimed ~2× the frame where
  it did, because rendered backdrops are smooth by construction. Diagnosed, quantified and
  documented rather than shipped.

**Project scope**
- Built Bricked, a full-stack photo-to-LEGO pipeline (FastAPI, MongoDB/GridFS, Motor, OpenCV,
  Open3D, SciPy, Ultralytics YOLO11, React 19, Vite, Tailwind, Three.js) that turns 8–16
  photos of an object into a buildable brick model and parts list.
- Implemented silhouette-based 3D reconstruction (visual hull / space carving over a 256³,
  16.7 M-voxel grid) from first principles in NumPy — camera model, projection, analytic
  focal length, chunked memory-bounded carving — with no calibration step.
- Designed a 5-stage async pipeline with a MongoDB state machine (409 on out-of-order stages,
  per-stage timestamps, failure capture) and GridFS storage for all binary/bulk artifacts.
- Built three Three.js viewers (point cloud, voxels, bricks) using `InstancedMesh` so a
  multi-hundred-brick model renders in one draw call, with leak-free WebGL teardown.

**Accuracy**
- Raised 3D reconstruction accuracy 23% (0.398 → 0.491 mean hull IoU) on a 19-object
  ground-truth benchmark; objects above 0.6 IoU went from 3 to 7, above 0.5 from 6 to 12.
- Cut colour error 37% (ΔE 20.2 → 12.8) by diagnosing three compounding defects from a real
  run's data: a saturation boost applied twice with an unconditional floor lift that gave
  zero-saturation pixels a hue, no white-balance stage at all, and a nearest-colour search
  that weighted lightness equally with chroma so neutral grey matched Dark Tan over Dark
  Stone Gray.
- Estimated the illuminant from the backdrop rather than the object, after measuring that an
  object-based estimate neutralises a genuinely monochrome object — a red object leads any
  grey-world estimator to conclude the light is red.
- Expanded the LEGO palette 19 → 52 real colours, scoring candidates against the existing
  set and rejecting 5 as perceptually redundant; the neutral ramp went from 4 lightness
  levels to 7, cutting grey-ramp quantization error 26%.
- Fixed silhouette-scale, camera-axis and view-bearing bugs in the carver and added
  two-ring (level + overhead) capture support; hull IoU on 16-photo captures 0.353 → 0.484.
- Identified that per-image mask IoU on synthetic backdrops was selecting the wrong
  segmenter, and replaced it with cross-view consistency measured on real photographs:
  mask-area spread across views cut 12× (std dev 5.0 → 0.4), the property silhouette
  intersection actually depends on.

**Performance**
- Cut upload-to-model latency 7.8× (72.6 s → 9.3 s) and worst case 9.6× (139.6 s → 14.5 s)
  on CPU.
- Reduced per-run storage 86× (541 MB → 6.3 MB) with surface-only point clouds, and point-cloud
  API latency 52× (10.5 s → 0.20 s).
- Sped up voxelization 72× (22.2 s → 0.31 s) and reconstruction 2.8× via early-exit carving.

**Buildability**
- Rewrote the brick packer (both orientations, colour-wildcard interiors, bond-aware
  staggered layers): 27% more studs per brick and bonded bricks up from 38% to 69%.
- Made models physically proportioned (1.2:1 brick height) and consistently sized
  (28-stud span); expanded the palette to 36 real LEGO colours, cutting colour error 15%.

**Quality and tooling**
- Built a benchmark suite: headless turntable renderer with exact ground truth, per-stage
  evaluator (IoU, structure, colour, timing), HTTP end-to-end harness and before/after
  reporting; used it to find and fix 11 defects and to tune 7 parameters by ablation.
- Added 12 unit tests covering camera geometry, pose assignment, carving accuracy,
  segmentation, voxelization and packing invariants.
- Fixed a MongoDB TLS configuration bug that blocked all local-database deployments.

---

## Scope of measurement

What this harness establishes, and what it leaves to real captures:

- **Measured against exact ground truth:** hull and voxel IoU, colour error, brick counts,
  studs per brick, structural bonding, connectivity, per-stage latency, payload and storage
  size. Renders supply a 64³ solid occupancy grid and per-photo masks, so these carry no
  labelling error.
- **Established by separate measurement on real photographs:** segmentation behaviour, in
  [Segmenter selection](#segmenter-selection). Synthetic backdrops are smooth by
  construction, so per-image mask scores on them do not predict real-capture behaviour; the
  cross-view consistency figures do.
- **Not modelled by the renderer:** lens distortion, motion blur and camera drift. Captures
  do include backdrop gradients, vignetting, per-shot tint and sensor noise.
- **Two-ring captures** score level with single-ring on this set, for the reason given above:
  synthetic objects are fully visible from one ring, so the benchmark cannot reward the
  vertical constraint the second ring supplies on real captures.

Every figure in this document is reproducible from `benchmark/` against the committed result
JSON; nothing is estimated or extrapolated.
