# Bricked

**Flick it & Brick it.** — Turn photos of a real object into a buildable LEGO model.

Upload a handful of photos taken from around an object. Bricked segments the object out of
each shot, reconstructs its 3D shape from the silhouettes, simplifies that shape into a
coarse voxel grid, packs the voxels into real LEGO brick footprints in real LEGO colors, and
gives you an interactive 3D model plus a parts list you could actually build from.

Every intermediate stage — masks, point cloud, voxel grid, brick layout — is rendered live in
the browser with Three.js as the pipeline runs.

On a 19-object ground-truth benchmark, the pipeline segments 100% of photos (mask IoU 0.98),
reconstructs shapes at 0.69 3D IoU, and goes from upload to finished brick model in 8.7 s
through the API — see [Benchmarks](#benchmarks),
[`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the full results and
[`docs/BRICKED_OVERVIEW.md`](docs/BRICKED_OVERVIEW.md) for a technical deep dive.

---

## How it works

| # | Stage | What actually happens |
|---|-------|----------------------|
| 1 | **Upload** | Images are validated, HEIC/HEIF is transcoded to JPEG, and everything is stored in GridFS |
| 2 | **Segment** | **YOLO11x-seg** instance segmentation with CLAHE + unsharp pre-processing and a low-confidence retry. Detections whose boxes touch the primary one are merged, so a part reported separately (a straw in a cup, a handle) is kept. Masks are gated on bounding-box fill ratio, and a rejected photo leaves a gap in the bearings rather than shifting later views |
| 3 | **Reconstruct** | Silhouette-based **visual hull carving** (space carving) over a 256³ grid, using an assumed turntable camera model. All silhouettes are cropped at one shared scale, each photo keeps the bearing of its position in the capture, and voxels are dropped as soon as a view rules them out. Colour is sampled from the photos; only the hull surface is stored |
| 4 | **Voxelize** | Fill the hull solid, then Open3D voxelization in **brick proportions** (1 stud wide, 1 brick = 1.2 studs tall) sized so the longest side is 28 studs, light Gaussian simplification, and connected-component cleanup |
| 5 | **LEGO Convert** | Colours quantized in CIE-LAB (k-means → a 36-colour palette of real LEGO colours), then greedy brick packing in both orientations. Hidden interior cells are colour wildcards, and layers alternate direction with a bonus for bridging bricks below, so the model is built in a running bond |
| 6 | **Visualize** | Three.js renders the point cloud, voxel grid, and final brick model; the parts list shows counts per brick type and color |

> **Note on reconstruction:** this is silhouette-based space carving, **not**
> Structure-from-Motion. There is no feature matching or bundle adjustment. Cameras are
> assumed to be evenly spaced in azimuth on a circle around the object at a slight downward
> elevation (~17°), which is why capture technique matters (see below). An even upload of
> 12 or more photos is read as two rings: the first half level, the second half from ~45°
> above at the same bearings.

### Getting good results

Photograph the object against a plain, contrasting background in even light, with the
camera held still. The recommended capture is **8 shots at eye level, turning the object
45° clockwise (seen from above) each time**, optionally followed by **8 more from above at
the same bearings**. Upload the photos in the order they were taken — each photo's position
sets its assumed bearing. Images that fail segmentation are skipped automatically with a
reason shown in the UI; the others keep their bearings, and the run continues as long as at
least two images survive.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| Database | MongoDB (Atlas or local) + GridFS |
| Python DB driver | Motor (async PyMongo) |
| Segmentation | Ultralytics YOLO11x-seg (+ OpenCV CLAHE / unsharp pre-processing) |
| 3D reconstruction | OpenCV + NumPy (hand-written visual hull carving) |
| Shape simplification | Open3D + SciPy (`ndimage`) |
| Color quantization | OpenCV (CIE-LAB) + SciPy (k-means) |
| Brick packing | Pure Python / NumPy greedy packer |
| Frontend | React 19 + Vite + Tailwind CSS |
| 3D rendering | Three.js + OrbitControls |

---

## Backend setup

### Prerequisites

- **Python 3.12** — Open3D does not publish wheels for 3.13+
- MongoDB Atlas cluster, or local MongoDB 6+
- On headless Linux, Open3D needs a few system libraries:
  `apt-get install libegl1 libgl1 libgomp1 libusb-1.0-0`

### Install

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

YOLO weights (`yolo11x-seg.pt`, ~120 MB) are downloaded automatically by Ultralytics the
first time the YOLO fallback runs, and are gitignored.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://localhost:27017` | Connection string (use the full `mongodb+srv://` URI for Atlas) |
| `MONGO_DB` | `bricked` | Database name |

### Run

```bash
cd backend
source venv/bin/activate
MONGO_URI="mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority" \
  uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`. Interactive API docs at `/docs`.

> **Atlas TLS:** for `mongodb+srv://` (or `tls=true`) URIs the Motor client passes `certifi`'s
> CA bundle as `tlsCAFile`, which fixes the SSL verification failures Atlas throws on systems
> without a usable system CA store (commonly macOS). Plain local URIs connect without TLS.

---

## Frontend setup

### Prerequisites

- Node.js 18+

### Install and run

```bash
cd frontend
npm install
npm run dev
```

App runs at `http://127.0.0.1:5173`. The backend's CORS policy allows exactly
`http://localhost:5173` and `http://127.0.0.1:5173`.

If your backend is not on `http://localhost:8000`, create `frontend/.env.local`:

```
VITE_API_BASE_URL=http://localhost:8000
```

### Production build

```bash
npm run build     # output in frontend/dist/
```

---

## API

All routes are mounted under `/api`. The pipeline is linear — trigger each stage in order.
Each stage validates the run's current status and returns **409** if called out of sequence.

```
uploaded → segmenting → segmented → reconstructing → reconstructed
         → voxelizing → voxelized → converting → complete
                                                     └─ failed (from any stage)
```

### Upload & storage

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/uploads/runs` | Create a run. `multipart/form-data`, field `images`, 1–20 files (the UI enforces 4–16). Max 50 MB each; JPEG / PNG / WebP / HEIC / HEIF |
| `GET` | `/api/uploads/runs/{run_id}` | Full run document |
| `GET` | `/api/uploads/images/{file_id}` | Stream a stored image from GridFS |

### Pipeline triggers

All return `202`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/runs/{run_id}/segment` | YOLO segmentation + mask quality gating |
| `POST` | `/api/runs/{run_id}/reconstruct` | Visual hull carving → colored point cloud |
| `POST` | `/api/runs/{run_id}/voxelize` | Open3D voxelization + shape simplification |
| `POST` | `/api/runs/{run_id}/lego` | Color quantization + brick packing + parts list |

### Results

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/runs/{run_id}/status` | Current status and last error |
| `GET` | `/api/runs/{run_id}/model` | Full brick layout and stud dimensions (requires `complete`) |
| `GET` | `/api/runs/{run_id}/parts` | Parts list and brick count (requires `complete`) |
| `GET` | `/api/runs/{run_id}/pointcloud` | Point cloud, sub-sampled to 8,000 points |
| `GET` | `/api/runs/{run_id}/voxels` | Voxel grid, sub-sampled to 12,000 voxels |
| `GET` | `/` | Health check |

Visualization endpoints stride-sample so browser payloads stay bounded regardless of grid
resolution.

**Example `/parts` response**

```json
{
  "run_id": "abc123",
  "brick_count": 412,
  "parts": [
    { "type": "2x4", "color_name": "Bright Red", "color": "#C91A09", "count": 63 },
    { "type": "1x2", "color_name": "White",      "color": "#FFFFFF", "count": 118 }
  ]
}
```

**Example `/model` response**

```json
{
  "bricks": [
    { "x": 3, "y": 0, "z": 5, "width": 2, "depth": 4, "height": 1,
      "type": "2x4", "color": "#C91A09", "color_name": "Bright Red" }
  ],
  "dimensions": { "width": 29, "height": 24, "depth": 27 }
}
```

Supported brick footprints: `2x4`, `2x3`, `2x2`, `1x4`, `1x3`, `1x2`, `1x1`, in either
orientation. A footprint is only accepted when every *visible* cell under it shares the same
quantized LEGO colour, so a brick never shows two colours; hidden interior cells match
anything. Brick coordinates are in studs horizontally and brick heights vertically.

---

## Benchmarks

Accuracy is measured against ground truth on **19 textured 3D models** from the
[Khronos glTF sample assets](https://github.com/KhronosGroup/glTF-Sample-Assets) (a fox,
chairs, sofas, a helmet, a lantern, a bottle, a truck, …). `benchmark/render_turntable.py`
renders each one as a turntable capture that matches the backend's camera model: 960×720
JPEGs on a studio backdrop with a lighting gradient, vignette, random tint and sensor
noise, plus ground-truth masks and a 64³ solid occupancy grid. `benchmark/evaluate.py` then
runs the backend's own stage functions on every capture. "Before" is the same harness run
on the code as it stood before this work; "Now" is current `master`. Timings are
single-process CPU wall-clock.

> The harness measures geometry, colour, packing and performance against exact ground truth.
> Segmentation is evaluated separately on real photographs, since rendered backdrops are
> smooth by construction — see
> [Segmenter selection](docs/BENCHMARK.md#segmenter-selection) for that measurement and the
> full result tables.

**8-photo capture (one ring):**

| Metric | Before | After |
| --- | --- | --- |
| Captures reconstructed | 18 / 19 | 18 / 19 |
| Photos segmented | 72% | 76% |
| Segmentation mask IoU | 0.779 | 0.854 |
| 3D IoU — carved hull (mean) | 0.398 | 0.491 |
| 3D IoU — final voxel model | 0.374 | 0.458 |
| Bricks per model | 2,861 | 1,075 |
| Studs per brick | 3.12 | 4.16 |
| Bricks bonded to ≥ 2 bricks below | 38% | 75% |
| Colour error, ΔE (visible surface) | 20.2 | 12.8 |
| Pipeline compute time per model | 43.2 s | 9.4 s |

**16-photo capture (level ring + high ring, as the UI recommends):**

| Metric | Before | After |
| --- | --- | --- |
| Photos segmented | 72% | 77% |
| 3D IoU — carved hull (mean) | 0.353 | 0.484 |
| 3D IoU — final voxel model | 0.336 | 0.451 |
| Bricks per model | 2,649 | 1,057 |
| Bricks bonded to ≥ 2 bricks below | 36% | 74% |
| Pipeline compute time per model | 56.1 s | 17.2 s |

**Through the HTTP API** (`benchmark/e2e_api.py`: FastAPI + local MongoDB/GridFS, 8 photos,
all 19 models, mean per run):

| Metric | Before | After |
| --- | --- | --- |
| Runs completed | 18 / 19 | 18 / 19 |
| Upload → finished model | 72.6 s | 9.3 s |
| Point cloud stored in GridFS | 541 MB (5.0 M points) | 7.5 MB (107 k points) |
| `GET /pointcloud` latency | 10.5 s | 0.15 s |
| `GET /model` payload | 318 kB | 121 kB |

Per stage (8 photos): segmentation 9.7 s → 5.6 s, reconstruction 11.1 s → 3.4 s,
voxelization 22.2 s → 0.3 s. Brick counts are not directly comparable — the old grid size
depended on photo framing, the new one is fixed at 28 studs — so studs per brick is the
fairer packing measure. One model (ToyCar) was dropped because its draped cloth cannot be
solid-filled into a reliable ground truth.

What changed, and why each fix was needed:

- **Segmentation** — only the single largest mask used to be kept, so a part the detector
  reported separately (a straw in a cup, a handle) was amputated before reconstruction.
  Detections whose boxes touch the primary one are now merged.
- **Colour** — saturation was boosted twice, once with an unconditional floor lift that gave
  zero-saturation pixels a hue, and there was no white balance, so a grey-and-white object
  photographed under warm light quantized to tans and browns. Gains are now estimated from
  the backdrop, neutral samples match only neutral bricks, and the palette carries 52 colours
  with a 7-level neutral ramp. ΔE 20.2 → 12.8.
- **Consistent silhouette scale** — each silhouette used to be cropped to its own bounding
  box and stretched to fill the frame, so a fox seen end-on was scaled up to the size of its
  side view. All views now share one crop.
- **Camera axes** — the projection treated image *y* as pointing up, so photos were carved
  as if upside-down relative to the camera tilt.
- **Bearings survive skipped photos** — a rejected photo used to shift every later photo
  onto the wrong angle; each photo now keeps the bearing of its position in the upload.
- **Two-ring captures** — the UI asked for 8 level + 8 high shots, but the backend spread
  all 16 around one level circle.
- **Carving** — strict silhouette intersection with no thin-part opening (tuned on the
  benchmark), and voxels are discarded as soon as one view rules them out: 2.9× faster.
- **Voxelization** — `OPEN_ITERS = 0` made SciPy erode the whole grid away on every run
  (it silently fell back to the raw grid); outlier removal stripped real thin geometry;
  cubic voxels built a model 20% too tall. Only the hull surface is now stored (86× smaller
  point cloud), and the grid is filled, sized in studs, and shaped like real bricks.
- **Brick packing** — footprints are tried in both orientations, hidden interior cells don't
  have to match colours, and layers are staggered, giving fewer, bigger bricks in a running
  bond instead of stacked columns.
- **Local MongoDB** — `tlsCAFile` was always passed, which forces TLS, so the documented
  default `mongodb://localhost:27017` could never connect.

Reproduce (needs `pyrender`, `trimesh` and an OSMesa build of Mesa for rendering):

```bash
PYOPENGL_PLATFORM=osmesa python benchmark/render_turntable.py models/*.glb --out scenes
PYOPENGL_PLATFORM=osmesa python benchmark/render_turntable.py models/*.glb --out scenes16 --rings 0.3 0.8
cd backend && python ../benchmark/evaluate.py ../scenes --json results.json
python ../benchmark/e2e_api.py ../scenes --json e2e.json      # with uvicorn + MongoDB running
```

Raw results for the tables above are in `benchmark/results/` (`orig*` = before, `new*` = after).

---

## Testing

```bash
cd backend
pip install pytest
python -m pytest -q        # 12 tests, no MongoDB or YOLO weights needed
```

The tests cover the camera convention, bearing assignment (gaps and two-ring captures),
shared-scale cropping, hull accuracy on an analytic box, backdrop segmentation (gradients,
through-holes, busy backgrounds), interior filling and stud sizing, palette consistency, and
that brick packing covers every voxel exactly once.

---

## Debug tooling

`backend/visualize_pointcloud.py` renders a run's reconstructed point cloud as a matplotlib
3D scatter, straight from MongoDB — useful for inspecting reconstruction quality without the
web app.

```bash
python visualize_pointcloud.py <run_id>
python visualize_pointcloud.py <run_id> --save output.png
python visualize_pointcloud.py <run_id> --mongo-uri mongodb://localhost:27017 --db bricked
```

---

## Project structure

```
Bricked/
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI app, CORS, lifespan hooks
│   │   ├── config.py                      # Env config, size/MIME limits
│   │   ├── database.py                    # Motor client + GridFS bucket (certifi TLS for Atlas)
│   │   ├── routers/
│   │   │   ├── uploads.py                 # Upload, fetch run, stream image
│   │   │   ├── pipeline.py                # Segment / reconstruct / voxelize / lego triggers
│   │   │   └── model.py                   # Model, parts, pointcloud, voxels, status
│   │   └── services/
│   │       ├── upload_service.py          # Validation, HEIC transcode, GridFS storage
│   │       ├── segmentation_service.py    # YOLO11x-seg + mask merge, quality gating
│   │       ├── reconstruction_service.py  # Visual hull carving + color sampling
│   │       ├── voxel_service.py           # Open3D voxelization + simplification
│   │       └── lego_service.py            # Color quantization + brick packing
│   ├── tests/test_pipeline.py             # Unit tests (no MongoDB or YOLO needed)
│   ├── visualize_pointcloud.py            # Standalone point cloud debug viewer
│   └── requirements.txt
├── benchmark/
│   ├── render_turntable.py                # Synthetic turntable captures + ground truth
│   ├── evaluate.py                        # Accuracy / structure / timing per stage
│   ├── e2e_api.py                         # End-to-end latency through the HTTP API
│   └── report.py                          # Before/after markdown table
└── frontend/
    ├── index.html
    ├── public/                            # Theme imagery
    ├── src/
    │   ├── main.jsx                       # All React components + Three.js viewers
    │   └── styles.css                     # Tailwind layers + dark root theme
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

Run state lives in the `runs` collection; all binaries and bulk JSON (originals, segmented
PNGs, point clouds, voxel grids, brick layouts) live in GridFS, referenced by file ID.

---

## Current limitations

- The pipeline runs **synchronously inside the HTTP request** — no background job queue — so a
  long reconstruction holds the connection open for its duration.
- **No authentication or rate limiting.** Runs are addressable by anyone who has the ID.
- Reconstruction quality depends on the **turntable camera assumption** holding: evenly spaced
  photos at a consistent height. Irregular capture degrades the hull.
- Visual hull cannot recover concavities that are never visible against the background
  (the inside of a bowl, a watch strap's loop seen only edge-on).
- The turntable direction is assumed (object turned clockwise seen from above); turning it
  the other way reconstructs a mirror image.
- The accuracy benchmark uses rendered captures with exact ground truth; real photos add
  lens distortion, shadows and camera drift that it does not model.
- **No CI** yet — tests and the benchmark are run by hand.
