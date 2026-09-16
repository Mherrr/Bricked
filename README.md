# Bricked

**Flick it & Brick it.** — Turn photos of a real object into a buildable LEGO model.

Upload a handful of photos taken from around an object. Bricked segments the object out of
each shot, reconstructs its 3D shape from the silhouettes, simplifies that shape into a
coarse voxel grid, packs the voxels into real LEGO brick footprints in real LEGO colors, and
gives you an interactive 3D model plus a parts list you could actually build from.

Every intermediate stage — masks, point cloud, voxel grid, brick layout — is rendered live in
the browser with Three.js as the pipeline runs.

---

## How it works

| # | Stage | What actually happens |
|---|-------|----------------------|
| 1 | **Upload** | Images are validated, HEIC/HEIF is transcoded to JPEG, and everything is stored in GridFS |
| 2 | **Segment** | CLAHE + unsharp pre-processing, then YOLO11x-seg instance segmentation; each mask is quality-gated by its bounding-box fill ratio |
| 3 | **Reconstruct** | Silhouette-based **visual hull carving** (space carving) over a 256³ grid, using an assumed turntable camera model, with per-voxel color sampled from the original photos |
| 4 | **Voxelize** | Open3D voxelization down to a coarse grid, then Gaussian-blur shape simplification and connected-component cleanup |
| 5 | **LEGO Convert** | Colors quantized to a 19-color LEGO palette in CIE-LAB via k-means, then color-aware greedy brick packing into standard footprints |
| 6 | **Visualize** | Three.js renders the point cloud, voxel grid, and final brick model; the parts list shows counts per brick type and color |

> **Note on reconstruction:** this is silhouette-based space carving, **not**
> Structure-from-Motion. There is no feature matching or bundle adjustment. Cameras are
> assumed to be evenly spaced in azimuth on a circle around the object at a slight downward
> elevation, which is why capture technique matters (see below).

### Getting good results

Photograph the object against a plain, contrasting background in even light. The recommended
capture is **8 shots at eye level, rotating the object ~45° each time, then 8 more from
above at the same bearings**. Images whose masks fail the quality gate are skipped
automatically with a reason shown in the UI — the run continues as long as at least two
images survive.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| Database | MongoDB (Atlas or local) + GridFS |
| Python DB driver | Motor (async PyMongo) |
| Segmentation | Ultralytics YOLO11x-seg |
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

### Install

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

YOLO weights (`yolo11x-seg.pt`, ~120 MB) are downloaded automatically by Ultralytics on the
first segmentation run and are gitignored.

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

> **Atlas TLS:** the Motor client passes `certifi`'s CA bundle as `tlsCAFile`, which fixes the
> SSL verification failures Atlas throws on systems without a usable system CA store (commonly
> macOS). No extra setup needed — `certifi` ships in `requirements.txt`.

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

Supported brick footprints: `2x4`, `2x3`, `2x2`, `1x4`, `1x3`, `1x2`, `1x1` — tried
largest-first, and only accepted when every cell under the footprint shares the same
quantized LEGO color, so a brick is never split across two colors.

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
│   │   ├── database.py                    # Motor client + GridFS bucket (certifi TLS)
│   │   ├── routers/
│   │   │   ├── uploads.py                 # Upload, fetch run, stream image
│   │   │   ├── pipeline.py                # Segment / reconstruct / voxelize / lego triggers
│   │   │   └── model.py                   # Model, parts, pointcloud, voxels, status
│   │   └── services/
│   │       ├── upload_service.py          # Validation, HEIC transcode, GridFS storage
│   │       ├── segmentation_service.py    # YOLO inference + mask quality gating
│   │       ├── reconstruction_service.py  # Visual hull carving + color sampling
│   │       ├── voxel_service.py           # Open3D voxelization + simplification
│   │       └── lego_service.py            # Color quantization + brick packing
│   ├── visualize_pointcloud.py            # Standalone point cloud debug viewer
│   └── requirements.txt
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
- Visual hull cannot recover concavities that are never visible against the background. A
  concavity-veto heuristic mitigates the worst cases but does not eliminate the limitation.
- **No tests or CI.**
- `trimesh` is listed in `requirements.txt` but is not used on the active code path; mesh/GLB
  export exists only as a commented-out sketch in `lego_service.py`. The model is served as
  JSON.
