# Bricked — Technical Overview

> Derived from the source code in this repository as of 2026-09-24 (branch `master`).
> Where the code and `README.md` disagree, the code is authoritative. All performance and
> accuracy figures come from the measured benchmark in `benchmark/` — see
> [`BENCHMARK.md`](BENCHMARK.md) for the full tables.

---

## What it is

Bricked is a full-stack web application that converts ordinary photographs of a physical
object into a buildable LEGO model.

A user uploads 4–16 photos taken while turning the object on a plain backdrop. The backend
segments the object out of each photo, carves a 3D shape from the silhouettes, converts that
shape into a brick-proportioned voxel grid, packs the voxels into real LEGO brick footprints
in real LEGO colours, and returns an interactive 3D model and a parts list. Every
intermediate stage — masks, point cloud, voxel grid, brick layout — is rendered live in the
browser as the pipeline progresses.

The frontend is themed as a pirate / deep-sea build console ("Flick it & Brick it").

**Headline results** (19-object ground-truth benchmark, 8-photo captures):

- 100% of photos segmented (was 72%), mask IoU 0.983 (was 0.779)
- 3D reconstruction IoU 0.689 (was 0.398) — **+73%**
- Upload → finished brick model through the API in **8.7 s** (was 72.6 s) — **8.3× faster**
- Point cloud stored per run **6.3 MB** (was 541 MB) — **86× smaller**
- 69% of bricks bonded to ≥ 2 bricks below (was 38%); 3.97 studs per brick (was 3.12)

**Scale:** ~3,600 lines of code — ~2,040 Python backend app, ~800 JSX frontend, ~560
Python benchmark suite, ~170 Python unit tests — plus config. Initial build over two days
(27 commits, 2026-04-18 → 2026-04-19, feature branches merged via pull request), followed
by a benchmark-driven accuracy and performance pass (2026-09).

---

## Architecture

```
┌──────────────────────┐        HTTP         ┌──────────────────────┐
│  React 19 SPA        │ ──────────────────> │  FastAPI (async)     │
│  Vite · Tailwind     │ <────────────────── │  5 service modules   │
│  Three.js viewers    │       JSON          │  3 routers           │
└──────────────────────┘                     └──────────┬───────────┘
                                                        │ Motor (async)
                                                        v
                                             ┌──────────────────────┐
                                             │  MongoDB             │
                                             │  · runs collection   │
                                             │  · GridFS (blobs)    │
                                             └──────────────────────┘
```

Two tiers, no job queue, no authentication. Each pipeline stage is its own HTTP endpoint;
the client triggers them in sequence and fetches visualization data between stages so the
UI fills in progressively rather than blocking on one long request.

### Repository layout

```
backend/
├── app/
│   ├── main.py                       FastAPI app, CORS, async lifespan DB hooks
│   ├── config.py                     MONGO_URI, MONGO_DB, size/MIME limits
│   ├── database.py                   Motor client + GridFS bucket (certifi TLS for Atlas only)
│   ├── routers/
│   │   ├── uploads.py                upload, fetch run, stream image
│   │   ├── pipeline.py               segment / reconstruct / voxelize / lego triggers
│   │   └── model.py                  model, parts, pointcloud, voxels, status
│   └── services/
│       ├── upload_service.py         validation, HEIC transcode, GridFS storage
│       ├── segmentation_service.py   backdrop model + GrabCut, YOLO11x-seg fallback
│       ├── reconstruction_service.py visual hull carving, view poses, colour sampling
│       ├── voxel_service.py          stud-space voxelization + simplification
│       └── lego_service.py           colour quantization + bond-aware brick packing
├── tests/test_pipeline.py            12 unit tests (no MongoDB or YOLO needed)
├── visualize_pointcloud.py           standalone CLI debug viewer (matplotlib)
└── requirements.txt
benchmark/
├── render_turntable.py               synthetic turntable captures + ground truth
├── evaluate.py                       accuracy / structure / timing per stage
├── e2e_api.py                        end-to-end latency through the HTTP API
├── report.py                         before/after markdown table
└── results/                          raw JSON results (orig* = before, new* = after)
frontend/
├── src/main.jsx                      all components + three Three.js viewers
├── src/styles.css                    Tailwind layers + dark root theme
├── tailwind.config.js                custom `shadow-abyss`, `bg-scan-lines`
├── vite.config.js                    host 127.0.0.1, port 5173
└── public/                           theme imagery
docs/
├── BRICKED_OVERVIEW.md               this file
└── BENCHMARK.md                      benchmark results and achievements
```

---

## State model

MongoDB is the pipeline's state machine. A single `runs` document tracks a job through:

```
uploaded → segmenting → segmented → reconstructing → reconstructed
         → voxelizing → voxelized → converting → complete
                                                     └─ failed (from any stage)
```

Every stage:

1. Loads the run and **rejects with HTTP 409 if `status` is not the expected predecessor**,
   so stages cannot be run out of order or twice.
2. Writes an in-progress status and a `*_started_at` timestamp before doing work.
3. On success, writes the terminal status, a result sub-document, and `*_completed_at`.
4. On exception, writes `status: "failed"` plus the error string, then re-raises as HTTP 500.

Result endpoints gate on `status == "complete"` (or on the presence of the relevant
sub-document) and return 409/404 rather than partial data.

### `runs` document shape

| Field | Written by | Contents |
|---|---|---|
| `status`, `error` | every stage | current state, last failure message |
| `created_at` | upload | UTC timestamp |
| `images[]` | upload | `file_id`, `filename`, `original_filename`, `size`, `content_type` |
| `segmented_images[]` | segment | `view_index` (position in the capture), `original_file_id`, `segmented_file_id`, `filename`, `detection{method, fill_ratio, confidence, box}` |
| `skipped_images[]` | segment | `filename`, human-readable `reason` |
| `reconstruction` | reconstruct | `point_cloud_file_id`, `point_count`, `method`, `grid_size` |
| `voxelization` | voxelize | `voxel_file_id`, `voxel_count`, `voxel_size`, `stud_span` |
| `lego` | lego | `model_file_id`, `brick_count`, `parts_list[]` |
| `*_started_at` / `*_completed_at` | each stage | per-stage timing |

Binary and bulk-JSON artifacts never live in the document — originals, segmented RGBA PNGs,
`point_cloud.json`, `voxels.json`, and `model.json` all go to **GridFS**, and the document
holds only file IDs and metadata.

---

## The pipeline

### 1 — Upload (`upload_service.py`)

Accepts 1–20 files per request (the UI enforces 4–16). Validates against a MIME allowlist
(`jpeg`, `png`, `webp`, `heic`, `heif`) and a 50 MB per-file cap.

**HEIC handling.** Browsers frequently send iPhone photos as `application/octet-stream`, so
content-type detection falls back to file-extension sniffing. Any HEIC/HEIF input is
transcoded to JPEG at quality 95 via `pillow-heif` *before* storage, so no downstream stage
ever receives a format OpenCV cannot decode. The stored filename is rewritten to `.jpg` and
both the original and canonical names are retained.

The run document is inserted first to obtain an ID, then each image is stored to GridFS
tagged with that `run_id`, then the document is updated with the image list. Upload order
is preserved and later determines each photo's assumed camera bearing.

### 2 — Segmentation (`segmentation_service.py`)

Two segmenters, chosen per image. All inference runs in a thread executor so the event
loop is never blocked.

**Primary: class-agnostic backdrop segmentation.** The app asks for a plain backdrop, so the
backdrop itself is modelled instead of the object:

1. The image is downscaled to 640 px (long side) and converted to CIE-LAB.
2. A **quadratic colour surface** (6-term basis: 1, x, y, x², y², xy) is least-squares fitted
   per LAB channel to a 4% border strip. The quadratic absorbs lighting gradients and
   vignetting that a single background colour cannot.
3. The border residual's robust spread (median absolute deviation) decides whether the
   backdrop is plain enough to model; a busy border (MAD > 4 LAB units, or > 10% of border
   pixels far off the model) hands the image to YOLO instead.
4. Foreground = residual above max(6 σ, 8 LAB units), cleaned with a 5×5 open/close.
5. **GrabCut** refines the mask, seeded with sure-foreground (eroded mask), sure-background
   (low residual and the border) and probable regions between.
6. The main object is kept (largest component plus pieces ≥ 5% of it), and only holes
   smaller than 1% of the object are filled — genuine openings like a mug handle or a watch
   strap where the backdrop shows through are kept.
7. Masks covering < 0.5% or > 90% of the frame are rejected.

This handles any object — an avocado, a boom box, a lantern — rather than only the 80 COCO
classes YOLO knows, and runs in ~0.5 s per image on CPU.

**Fallback: Ultralytics YOLO11x-seg**, loaded once and cached with `@lru_cache(maxsize=1)`.
Input is pre-processed with CLAHE on the L channel (clip 2.5, 8×8 tiles) and an unsharp mask
(σ = 2.0, weights 1.4 / −0.4). Detection runs at `conf=0.35`, retrying at `conf=0.10`; the
largest mask is taken and gated on **bounding-box fill ratio ∈ [0.40, 0.98]** (below:
fragmented mask; above: mask flooded into the background).

Rejected images are **skipped, not fatal**, each with a human-readable reason surfaced in the
UI. The stage fails only if fewer than 2 images survive. Output is one RGBA PNG per surviving
image, with the mask as alpha, plus the image's `view_index`.

### 3 — Reconstruction (`reconstruction_service.py`)

Method recorded as `visual_hull`: **silhouette-based space carving**, not
Structure-from-Motion — no feature matching, essential-matrix estimation or bundle
adjustment. The carving, camera construction and projection are hand-written NumPy.

**Camera model.** A turntable: views on a circle of radius 3.0 around the Y axis looking at
the origin, at 0.3 rad (~17°) elevation. `_build_camera()` builds each `R|t` from a look-at
basis whose axes match pixel coordinates (camera +X = image right, +Y = image down),
swapping the up-vector near vertical views.

**View poses (`view_poses`).** Each photo's bearing comes from its **position in the
upload**, not its position among survivors — so a photo rejected by segmentation leaves a
gap instead of shifting every later photo onto the wrong angle. An even upload of **12 or
more** photos is read as **two rings**: the first half level (0.3 rad), the second half from
above (0.8 rad, ~45°) at the same bearings, matching the UI's "8 level + 8 from above"
capture instructions.

**Silhouette normalization (`_crop_windows`).** Every silhouette is resampled from a square
crop window to 512 px. The window is **shared across views** — the padded union of all
silhouette bounding boxes — so every view has the same pixel scale. (With a fixed camera and
a turntable the rotation axis stays at a fixed image column, which makes this exact.) If
image sizes differ, a shared-scale mode (common window size, per-view centre) is used
instead. The focal length is then derived analytically,
`f = size / (2 · (1 + 2·pad)) · distance`, so no camera intrinsics or calibration are needed.
Each silhouette also gets a concavity mask (inside its convex hull but outside the
silhouette) and a 2 px dilation as an error margin.

**Carving.** A **256³ grid — 16.7 M voxels** — is carved in **4 M-voxel chunks**, with world
coordinates generated lazily from flat indices. Views are applied one at a time to the
**surviving voxels only**: a voxel is discarded the moment it can no longer reach the vote
threshold, so after the first view or two most of the grid is gone and later projections
touch a small fraction of it. The benchmark-tuned rule is **strict silhouette intersection**
(a voxel must fall inside every view's silhouette), with a concavity veto (≥ 2 views)
available for tolerant thresholds. A 3D largest-connected-component filter removes satellite
blobs.

**Surface-only output.** The interior of the hull carries no colour information, so only the
**surface shell** (voxels with an empty 6-neighbour) is kept and serialized — the
voxelization stage re-fills the solid. This cut the stored point cloud from ~5.0 M to ~90 k
points per run.

**Colour sampling.** For each surface voxel, only the sample from the camera with the
**smallest camera-space depth** (the view most directly facing it) is kept, preventing
cross-view bleed (a pink back surface onto a white front). A final HSV saturation boost
(×1.8, +25) recovers colours washed out by flat indoor lighting.

**Tunables:** `GRID_SIZE=256`, `CAMERA_DISTANCE=3.0`, `CAMERA_ELEVATION=0.3`,
`HIGH_ELEVATION=0.8`, `TWO_RING_MIN=12`, `NORM_SIZE=512`, `NORM_PAD=0.10`, `DILATION_PX=2`,
`THIN_OPEN_PX=0`, `CONCAVITY_VETO=2`, `MIN_VOTE_FRAC=1.0`, `NORMALIZATION="shared_crop"`,
`_PROJ_CHUNK=4_000_000`.

### 4 — Voxelization (`voxel_service.py`)

Turns the hull surface into a solid, buildable grid in **stud space**.

1. **Brick proportions.** Y is divided by 1.2 before voxelizing, because a LEGO brick is
   1.2× as tall as its stud pitch — so each cell is one stud wide and one brick tall and
   the built model keeps the object's real proportions (cubic cells made models 20% too
   tall).
2. **Size in studs.** The voxel pitch is chosen so the object's longest side is
   **28 studs**, independent of how the photos were framed.
3. **Open3D voxelization** (`VoxelGrid.create_from_point_cloud`) with averaged colours, then
   an HSV ×1.5 saturation boost to undo the grey-pull of colour averaging.
4. **Solid fill** (`binary_fill_holes`) of the surface shell.
5. **Gaussian smoothing** of the occupancy field (σ = 0.2) re-binarized at 0.45 — a shape
   simplifier expressed as a filter.
6. Optional morphological opening (`OPEN_ITERS`, off by default and skipped entirely at 0).
7. **Largest connected component**, with a fallback to the unsimplified grid.
8. **Colour assignment by nearest coloured cell**: a Euclidean distance transform with
   `return_indices` gives every voxel — including interior cells — the colour of its nearest
   surface cell in one pass.

No statistical outlier removal: the hull is already a single connected component, and
outlier removal was measured to strip real thin geometry (voxel IoU 0.587 → 0.616 without
it).

**Tunables:** `TARGET_STUDS=28`, `BRICK_ASPECT=1.2`, `GAUSS_SIGMA=0.2`, `GAUSS_THRESH=0.45`,
`OPEN_ITERS=0`.

### 5 — LEGO conversion (`lego_service.py`)

**Palette.** **36 real LEGO colours** with Rebrickable hex values — reds, blues (incl. Dark
Blue, Bright Light Blue, Sand Blue, Medium/Dark Azure, Light Aqua), yellows and oranges (incl.
Bright Light Orange, Dark Orange), greens (incl. Lime, Yellowish Green, Olive Green, Sand
Green), neutrals (White, Black, Medium/Dark Stone Gray), browns and skin tones (Reddish Brown,
Dark Brown, Nougat, Medium/Light Nougat, Tan, Dark Tan), and pinks/purples (Coral, Bright
Pink, Magenta, Bright Purple, Medium Lavender). Pre-converted to **CIE-LAB** at import for
perceptually uniform matching; a unit test checks every hex against its RGB triple.

**Dominant-palette derivation (k-means).** Colours of the **visible** voxels are converted to
LAB and clustered (**k = 6**, `scipy.cluster.vq`); only the cluster centroids are mapped to
LEGO colours, and every visible voxel is quantized against that restricted palette. Stray
noise pixels are absorbed into the dominant cluster instead of adding phantom colours.

**Bond-aware greedy packing.** Bricks are packed layer by layer in Y:

- Footprints `2x4 → 2x3 → 2x2 → 1x4 → 1x3 → 1x2 → 1x1` are tried **in both orientations**.
- A footprint is accepted only if every **visible** cell under it shares one quantized
  colour; **hidden interior cells are colour wildcards**, so the core packs into large
  bricks. Interior-only bricks take the model's most common colour.
- Candidates are scored `area + 3.0 × (distinct bricks bridged in the layer below)`, and
  even/odd layers alternate both preferred orientation and scan order, producing a
  **running bond** instead of stacked columns that would fall apart.

**Output.** A brick layout (`x, y, z, width, depth, height, type, color, color_name`;
coordinates in studs horizontally and brick heights vertically) plus overall dimensions,
stored as `model.json` in GridFS, and a parts list aggregated by `(type, color_name, color)`.

---

## API surface

All routes are mounted under `/api`. Pipeline triggers return `202`.

### Upload & storage

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/uploads/runs` | `multipart/form-data` field `images`, 1–20 files. Creates the run. |
| `GET` | `/api/uploads/runs/{run_id}` | Full run document. |
| `GET` | `/api/uploads/images/{file_id}` | Streams an image out of GridFS chunk-by-chunk. |

### Pipeline triggers

| Method | Path | Stage |
|---|---|---|
| `POST` | `/api/runs/{run_id}/segment` | Backdrop / YOLO segmentation |
| `POST` | `/api/runs/{run_id}/reconstruct` | Visual hull carving |
| `POST` | `/api/runs/{run_id}/voxelize` | Stud-space voxelization |
| `POST` | `/api/runs/{run_id}/lego` | Brick packing + parts list |

### Results

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/runs/{run_id}/status` | Current status + last error. |
| `GET` | `/api/runs/{run_id}/model` | Full brick layout + dimensions. Requires `complete`. |
| `GET` | `/api/runs/{run_id}/parts` | Parts list + brick count. Requires `complete`. |
| `GET` | `/api/runs/{run_id}/pointcloud` | Point cloud, **sub-sampled to 8,000 points**. |
| `GET` | `/api/runs/{run_id}/voxels` | Voxel grid (≤ 12,000 voxels) + `voxel_size`, `stud_span`. |
| `GET` | `/` | Health check. |

Every route validates the run ID as a BSON `ObjectId` (400 on a malformed one). Measured
latency: `GET /pointcloud` 0.20 s (was 10.5 s); `GET /model` payload 103 kB (was 318 kB).

---

## Frontend

**Stack:** React 19, Vite 7, Tailwind CSS 3, Three.js 0.184.

**Shared scene factory.** `makeScene()` builds every 3D view: a `WebGLRenderer` with alpha
and capped pixel ratio, a perspective camera, `OrbitControls` with damping and auto-rotate
that **stops permanently the first time the user grabs the scene**, a `ResizeObserver` that
keeps canvas and camera aspect in sync, and a `dispose()` teardown that cancels the animation
frame, disconnects the observer, removes the canvas and frees GPU resources — so repeated
re-renders never leak WebGL contexts.

**Three viewers:**

- **Point cloud** — `THREE.Points` over interleaved position/colour `BufferAttribute`s,
  sampled colours or an HSL height ramp; axes helper, origin marker, ground grid, legend.
- **Voxel grid** — one `InstancedMesh` of cells drawn at true brick height (1.2:1), with
  per-instance colours, auto-framed from the data's bounding box.
- **LEGO model** — one `InstancedMesh` where each instance's **scale encodes the brick
  footprint** and height is drawn at the real 1.2 brick-to-stud ratio, so a
  multi-hundred-brick model is a single draw call.

**Upload panel.** Click-or-drag-and-drop, thumbnail previews with per-file size, removal,
deduplication by `name-size`, `URL.revokeObjectURL` cleanup, live 4–16 validation, and
capture instructions (plain backdrop, still camera, turn 45° clockwise, 8 level + 8 from
above, upload in the order taken).

**Orchestration.** `uploadImages()` drives the pipeline client-side: upload → segment →
reconstruct → *fetch point cloud* → voxelize → *fetch voxels* → lego → *fetch model*, with a
five-step progress panel, non-fatal visualization fetches, and backend `detail` messages
surfaced to the user.

**Results panels.** Segmented masks in a scrollable grid with colour-coded fill-ratio badges,
skipped images with their reasons, then brick count, stud dimensions, the 3D model and a
parts list with colour swatches.

**Theme.** Custom Tailwind extensions (`shadow-abyss`, `scan-lines` CRT background), a dark
amber-on-near-black palette, and a pirate-voiced copy layer.

---

## Testing and benchmarking

**Unit tests** — `backend/tests/test_pipeline.py`, 12 tests, pytest, no MongoDB or YOLO
weights required. They cover: camera axes match pixel axes; bearings keep gaps for skipped
photos; two-ring pose assignment; shared-scale cropping; hull accuracy > 0.8 IoU on an
analytic elongated box; backdrop segmentation on a gradient backdrop (> 0.97 IoU); openings
preserved through a ring; busy backgrounds rejected as backdrops; hollow shells filled and
sized to 28 studs in brick proportions; palette hex/RGB consistency; and brick packing
covering every voxel exactly once.

**Benchmark suite** — `benchmark/`:

- `render_turntable.py` renders textured glTF models (headless pyrender/OSMesa) as turntable
  captures matching the backend's camera model: 960×720 JPEGs on a studio backdrop with a
  lighting gradient, vignette, random tint and sensor noise, plus ground-truth masks and a
  64³ solid occupancy grid (surface rasterized, closed, floor-sealed, flood-filled).
- `evaluate.py` runs the backend's own stage functions and scores segmentation (pass rate,
  mask IoU), reconstruction (3D IoU of hull and final voxels after bounding-box
  normalization), LEGO structure (studs per brick, share of bricks bonded to ≥ 2 below,
  largest connected build), colour error (CIE76 ΔE on visible voxels) and per-stage time.
- `e2e_api.py` drives a live FastAPI + MongoDB server over HTTP and records per-request
  latency, GridFS storage and payload sizes.
- `report.py` prints before/after tables; raw JSON results live in `benchmark/results/`.

---

## Tooling

`backend/visualize_pointcloud.py` is a standalone CLI debug viewer: it connects to MongoDB,
pulls a run's point cloud out of GridFS and renders it as a matplotlib 3D scatter.

```
python visualize_pointcloud.py <run_id> [--mongo-uri URI] [--db NAME]
                                        [--save out.png] [--max-points N]
```

---

## Dependencies

**Backend** (`requirements.txt`): FastAPI, Uvicorn (standard), Motor, PyMongo, certifi,
python-multipart, Pillow, pillow-heif, Ultralytics, opencv-python-headless, NumPy, SciPy,
Open3D, matplotlib. (Trimesh, previously declared but unused, was removed.)

`certifi` is passed as `tlsCAFile` **only for TLS URIs** (`mongodb+srv://`, `tls=true`,
`ssl=true`) — which fixes Atlas on systems without a usable CA store while letting a plain
local MongoDB connect. Open3D constrains the backend to Python 3.12 and, on headless Linux,
needs `libegl1 libgl1 libgomp1 libusb-1.0-0`.

**Frontend** (`package.json`): react, react-dom, three, vite, @vitejs/plugin-react,
tailwindcss, postcss, autoprefixer.

**Benchmark only:** pyrender, trimesh, PyOpenGL + OSMesa, httpx, pytest.

---

## Running it

**Backend** — from `backend/`, in a Python 3.12 virtualenv with `requirements.txt`:

```bash
uvicorn app.main:app --reload                                # local MongoDB
MONGO_URI="mongodb+srv://..." uvicorn app.main:app --reload  # Atlas
```

Defaults: `MONGO_URI=mongodb://localhost:27017`, `MONGO_DB=bricked`. Serves on
`http://localhost:8000`; OpenAPI docs at `/docs`. YOLO weights download on first fallback use.

**Frontend** — from `frontend/`: `npm install && npm run dev` → `http://127.0.0.1:5173`.
Override the backend with `VITE_API_BASE_URL` in `.env.local`. CORS allows exactly
`http://localhost:5173` and `http://127.0.0.1:5173`.

**Tests / benchmark:** `python -m pytest -q` in `backend/`; benchmark commands in
`README.md` → *Benchmarks*.

---

## Engineering notes worth calling out

- **3D vision implemented from first principles.** Carving, camera construction, projection
  and focal-length derivation are hand-written NumPy; view poses, shared-scale cropping and
  two-ring captures were added after benchmarking exposed inconsistent geometry.
- **Classical CV where it wins, deep learning as the fallback.** A class-agnostic backdrop
  model (quadratic LAB fit + GrabCut) replaced YOLO as the primary segmenter after YOLO was
  measured missing or rejecting 28% of photos; YOLO remains for cluttered backgrounds.
- **Measurement-driven engineering.** A ground-truth benchmark (19 models, 152 photos per
  capture set) was built first; every change was kept or rejected on measured IoU, brick
  structure and latency, including parameter sweeps over normalization mode, vote threshold,
  opening, dilation, smoothing, colour clusters and bond weight.
- **Colour handled as a real problem.** LAB for perceptual distance, HSV saturation recovery,
  depth-arbitrated multi-view sampling, k-means before quantization, a 36-colour real LEGO
  palette, and ΔE measured only on what is visible.
- **Physical buildability.** Brick-proportioned voxels (1.2:1), footprints in both
  orientations, colour-wildcard interiors and a bond-aware running-bond packer.
- **Memory and throughput.** Chunked traversal of a 16.7 M-voxel grid with lazily generated
  coordinates, early-exit carving over survivors only, surface-only storage (86× smaller),
  distance-transform colour fill, `InstancedMesh` batching, API response sub-sampling.
- **Async correctness.** Every CPU-bound stage runs via `run_in_executor`; DB lifecycle via
  FastAPI's lifespan; image responses stream instead of buffering.
- **Failure handling that degrades rather than collapses.** Per-image skipping with reasons,
  segmenter fallback, confidence retry, simplification fallbacks, non-fatal visualization
  fetches, and actionable error strings propagated to the UI.

---

## Bugs found and fixed by the benchmark

| # | Area | Problem | Fix |
|---|---|---|---|
| 1 | Segmentation | YOLO (80 COCO classes) missed or rejected 28% of photos; partial masks (mean IoU 0.78) | Backdrop model + GrabCut primary, YOLO fallback → 100%, IoU 0.983 |
| 2 | Reconstruction | Each silhouette cropped to its own box and stretched, so views had different scales | One shared crop window for all views |
| 3 | Reconstruction | Camera basis treated image *y* as up, so photos were carved upside-down relative to the tilt | Camera axes match pixel axes |
| 4 | Reconstruction | A skipped photo shifted every later photo to the wrong bearing | Bearing from upload position (`view_index`) |
| 5 | Reconstruction | UI's 16-photo two-ring capture was spread around one level circle | Two-ring pose model for even uploads ≥ 12 |
| 6 | Voxelization | `OPEN_ITERS = 0` made SciPy erode the whole grid away on every run (silent fallback) | Opening skipped when disabled |
| 7 | Voxelization | Statistical outlier removal stripped real thin parts | Removed (hull is already one component) |
| 8 | Voxelization | Cubic voxels built models 20% too tall; size depended on photo framing | Brick-proportioned cells, 28-stud target |
| 9 | LEGO | Footprints tried in one orientation only; interior colour noise fragmented bricks; identical layers stacked into columns | Both orientations, wildcard interiors, bond-aware staggered layers |
| 10 | Database | `tlsCAFile` always passed, forcing TLS, so the default local MongoDB could never connect | TLS only for TLS URIs |
| 11 | Storage | Full solid hull (~5 M points, 541 MB JSON) stored per run | Surface-only storage (6.3 MB) |

---

## Known quirks and rough edges

- **`elevation_rad`** is returned by `GET /pointcloud` but never written by the
  reconstruction stage, so it is always `null`.
- **`image_sizes`** is still threaded through `_visual_hull_carving()` but unused.
- **Image-count limits differ by layer**: backend 1–20, UI 4–16, one heading reads "8–16".
- **`GET /status` is unused by the frontend**, which tracks progress in React state.
- **The pipeline runs synchronously inside the request** — no background job queue.
- **No authentication, rate limiting, per-user scoping, CI or deployment config.**
- Reconstruction assumes the turntable model: photos evenly spaced, taken in order, object
  turned clockwise (seen from above). Turning it the other way yields a mirror image.
- Visual hull cannot recover concavities never seen against the background (inside of a
  bowl, a watch strap's loop seen edge-on).
- The accuracy benchmark is synthetic; real photos add lens distortion, shadows and camera
  drift, so real-world accuracy will be lower than the benchmark figures.
