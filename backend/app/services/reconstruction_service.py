"""
Silhouette-based 3D reconstruction (Visual Hull / Space Carving).

For each segmented RGBA image, the object silhouette is back-projected through
an assumed turntable camera. Voxels that project outside every silhouette are
carved away; the remainder forms the visual hull, stored as a point cloud.

Camera assumption: N cameras equally spaced in azimuth on a circle around the
Y-axis at a fixed radius and slight downward elevation — a common handheld
multi-photo setup.
"""

import asyncio
import io
import json
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
from bson import ObjectId
from fastapi import HTTPException
from scipy.ndimage import binary_erosion, generate_binary_structure, label as nd_label

from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

# ── Carving parameters ────────────────────────────────────────────────────────

GRID_SIZE        = 256   # Voxel grid resolution (N³)
CAMERA_DISTANCE  = 3.0   # Distance from camera to object origin; object spans [-1,1]³
CAMERA_ELEVATION = 0.3   # Radians (~17°) — cameras tilt slightly downward
HIGH_ELEVATION   = 0.8   # Radians (~45°) — the "from above" ring of a two-ring capture
TWO_RING_MIN     = 12    # an even upload of at least this many photos is two rings

# Silhouette normalisation — each raw mask is cropped to its bounding box,
# padded, and resized to a square before projection.  The focal length is then
# derived analytically so the ±1 world extent maps to the padded boundary.
# Rule: NORM_SIZE = 2 × GRID_SIZE so adjacent voxels project to distinct pixels.
NORM_SIZE        = 512   # Normalised silhouette side length in pixels
NORM_PAD         = 0.10  # Fractional padding added around the bounding box
DILATION_PX      = 2     # Extra dilation on the normalised mask (error margin)
THIN_OPEN_PX     = 0     # Opening radius to cut thin protrusions (0 = keep them; they are real geometry)

# Concavity-enhanced carving: voxels projecting into the "phantom" zone
# (inside silhouette convex hull but outside the silhouette itself) are
# counted as concavity hits. If a voxel accumulates this many hits across
# all views it is carved regardless of the vote threshold.
CONCAVITY_VETO   = 2

# Coords are generated lazily per chunk — no large pts_world allocation.
_PROJ_CHUNK      = 4_000_000

# A voxel is kept when it lies inside at least this fraction of views.
# Higher = tighter hull, lower = more tolerant of camera model errors.
MIN_VOTE_FRAC    = 1.0

# How silhouettes are cropped before projection — see _crop_windows.
NORMALIZATION    = "shared_crop"


# ── Core silhouette helpers ───────────────────────────────────────────────────

def _extract_silhouette(rgba_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode an RGBA PNG and return (binary_mask H×W uint8, (width, height))."""
    buf = np.frombuffer(rgba_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode segmented image")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = (alpha > 32).astype(np.uint8)
    else:
        # Fallback: any non-white pixel is foreground
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        _, mask = cv2.threshold(gray, 250, 1, cv2.THRESH_BINARY_INV)

    # Small morphological close to fill fringe holes from YOLO mask borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape
    return mask, (w, h)


def _build_camera(
    angle: float,
    elevation: float,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (R, t) for a turntable camera.
    Camera sits on a circle of given radius at the given elevation angle and
    looks at the world origin.  Convention: X_cam = R @ X_world + t, with
    camera +X pointing image-right and +Y image-down, matching pixel
    coordinates (u right, v down) so photos are not interpreted upside-down.
    """
    cam_pos = np.array([
        distance * np.cos(elevation) * np.sin(angle),
        distance * np.sin(elevation),
        distance * np.cos(elevation) * np.cos(angle),
    ])

    z_axis = -cam_pos / np.linalg.norm(cam_pos)          # points into scene
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z_axis, world_up)) > 0.99:              # near-vertical view
        world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(z_axis, world_up)                    # image right
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)                      # image down

    R = np.stack([x_axis, y_axis, z_axis], axis=0)        # (3, 3)
    t = -R @ cam_pos                                       # (3,)
    return R, t


# ── Visual Hull carving ───────────────────────────────────────────────────────

def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(r0, r1, c0, c1) inclusive bounding box of the foreground, or None if empty."""
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if not rows.size:
        return None
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _square_crop(
    centre_r: float, centre_c: float, side: int, shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Square window of `side` px centred on (centre_r, centre_c), as slice bounds (may exceed the image)."""
    r0 = int(round(centre_r - side / 2.0))
    c0 = int(round(centre_c - side / 2.0))
    return r0, r0 + side, c0, c0 + side


def _crop_windows(masks: list[np.ndarray], mode: str = None) -> list[tuple[int, int, int, int] | None]:
    """
    Choose the square window each silhouette is resampled from.

    The carving camera model maps a fixed world extent to the window, so every
    view must be cropped at the *same* pixel scale — otherwise a long object
    seen end-on is blown up to the same size as its side view and the hull is
    carved from mutually inconsistent silhouettes.

      shared_crop  — one window for every view: the padded union of all
                     silhouette boxes.  Exact for a fixed camera and a turntable
                     (the object's rotation axis stays at a fixed image column).
      shared_scale — common window size, but centred on each view's own box.
                     Tolerates a camera that drifts between shots.
      per_view     — each view cropped to its own box (legacy behaviour).
    """
    mode = mode or NORMALIZATION
    boxes = [_bbox(m) for m in masks]
    valid = [b for b in boxes if b is not None]
    if not valid:
        return [None] * len(masks)

    if mode == "shared_crop" and len({m.shape for m in masks}) != 1:
        mode = "shared_scale"                     # windows only comparable at one resolution

    windows: list[tuple[int, int, int, int] | None] = []
    if mode == "shared_crop":
        r0 = min(b[0] for b in valid); r1 = max(b[1] for b in valid)
        c0 = min(b[2] for b in valid); c1 = max(b[3] for b in valid)
        side = int(np.ceil(max(r1 - r0, c1 - c0) * (1.0 + 2.0 * NORM_PAD))) + 1
        win = _square_crop((r0 + r1) / 2.0, (c0 + c1) / 2.0, side, masks[0].shape)
        return [win if b is not None else None for b in boxes]

    if mode == "shared_scale":
        side = int(np.ceil(max(max(b[1] - b[0], b[3] - b[2]) for b in valid) * (1.0 + 2.0 * NORM_PAD))) + 1
    for b, m in zip(boxes, masks):
        if b is None:
            windows.append(None)
            continue
        own = int(np.ceil(max(b[1] - b[0], b[3] - b[2]) * (1.0 + 2.0 * NORM_PAD))) + 1
        windows.append(_square_crop((b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0,
                                    side if mode == "shared_scale" else own, m.shape))
    return windows


def _crop_padded(img: np.ndarray, win: tuple[int, int, int, int]) -> np.ndarray:
    """Slice a window out of img, zero-filling wherever it runs past the border."""
    r0, r1, c0, c1 = win
    h, w = img.shape[:2]
    out = np.zeros((r1 - r0, c1 - c0) + img.shape[2:], dtype=img.dtype)
    sr0, sr1 = max(r0, 0), min(r1, h)
    sc0, sc1 = max(c0, 0), min(c1, w)
    if sr1 > sr0 and sc1 > sc0:
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = img[sr0:sr1, sc0:sc1]
    return out


def _normalize_silhouette(
    mask: np.ndarray,
    window: tuple[int, int, int, int] | None,
    orig_bgr: np.ndarray | None = None,
    out_size: int    = NORM_SIZE,
    dilation_px: int = DILATION_PX,
    open_px: int     = THIN_OPEN_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Resample the silhouette from its square crop window to out_size², apply
    optional opening to remove thin protrusions, then dilate for tolerance.

    If orig_bgr is provided (same H×W as mask), applies the identical crop and
    resize to produce a normalised RGB colour image for colour sampling.

    Returns (silhouette, concavity_mask, normed_rgb_or_None).
    """
    if window is None:
        empty = np.zeros((out_size, out_size), dtype=np.uint8)
        return empty, empty, None

    cropped = _crop_padded(mask, window)
    normed = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    # Normalise colour image with the same crop before any morphological ops
    normed_rgb: np.ndarray | None = None
    if orig_bgr is not None and orig_bgr.shape[:2] == mask.shape:
        color_crop    = _crop_padded(orig_bgr, window)
        color_resized = cv2.resize(color_crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        normed_rgb    = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)

    # Remove thin protrusions (straws, stems) via opening + largest component
    if open_px > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1,) * 2)
        opened = cv2.morphologyEx(normed, cv2.MORPH_OPEN, k_open)
        if opened.any():
            n_comp, labels = cv2.connectedComponents(opened.astype(np.uint8))
            if n_comp > 1:
                sizes   = np.bincount(labels.ravel())[1:]
                largest = int(np.argmax(sizes)) + 1
                normed  = (labels == largest).astype(np.uint8)
            else:
                normed = opened

    # Compute concavity mask BEFORE dilation so the hull boundary is clean.
    # Concavity = pixels inside the silhouette's convex hull but outside the
    # silhouette itself — the "phantom" regions visual hull keeps by default
    # (e.g. the space between trunk and face, between arm and body).
    concavity = np.zeros_like(normed)
    contours, _ = cv2.findContours(normed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull_mask = np.zeros_like(normed)
        for cnt in contours:
            hull_pts = cv2.convexHull(cnt)
            cv2.fillConvexPoly(hull_mask, hull_pts, 1)
        concavity = ((hull_mask > 0) & (normed == 0)).astype(np.uint8)

    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1,) * 2)
        normed = cv2.dilate(normed, kernel)

    return normed, concavity, normed_rgb


def _sample_colors(
    occupied_pts: np.ndarray,
    view_data: list,
    f_eff: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Assign each occupied voxel the colour from the **nearest valid camera view**.

    For each voxel, iterate over all views that have a colour image and where the
    voxel projects inside the silhouette.  Keep only the sample from the view with
    the smallest camera-space z-depth (i.e. the camera most directly facing that
    voxel).  This prevents cross-view colour bleed: a white-front voxel picks the
    front camera (smallest z from front), not an average with the pink back.

    After sampling, saturation is boosted so that studio/diffuse lighting doesn't
    wash colours out to near-gray.

    Returns (M, 3) uint8 RGB array; voxels with no valid sample default to gray.
    """
    M = len(occupied_pts)
    views_with_color = [v for v in view_data if v[4] is not None]

    if not views_with_color:
        logger.warning("_sample_colors: no colour images available — returning gray")
        return np.full((M, 3), 128, dtype=np.uint8)

    best_z     = np.full(M, np.inf, dtype=np.float32)
    best_color = np.full((M, 3), np.nan, dtype=np.float32)

    for sil, _cav, R, t, normed_rgb in views_with_color:
        pts_cam  = occupied_pts @ R.T + t
        z        = pts_cam[:, 2]
        in_front = z > 1e-4

        safe_z = np.where(in_front, z, 1.0)
        px = f_eff * pts_cam[:, 0] / safe_z + cx
        py = f_eff * pts_cam[:, 1] / safe_z + cy

        px_i = np.round(px).astype(np.int32)
        py_i = np.round(py).astype(np.int32)

        in_bounds = in_front & (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h)
        valid     = np.where(in_bounds)[0]

        if not valid.size:
            continue

        in_sil  = sil[py_i[valid], px_i[valid]] > 0
        sil_idx = valid[in_sil]
        if not sil_idx.size:
            continue

        # Update only where this view is closer (smaller z-depth) than any prior
        closer = z[sil_idx] < best_z[sil_idx]
        upd    = sil_idx[closer]
        if upd.size:
            best_z[upd]     = z[upd]
            best_color[upd] = normed_rgb[py_i[upd], px_i[upd]].astype(np.float32)

    sampled = (~np.isnan(best_color).any(axis=1)).sum()
    logger.info("_sample_colors: %d / %d voxels got a colour sample", sampled, M)

    no_data = np.isnan(best_color).any(axis=1)
    best_color[no_data] = 128.0
    colors = np.clip(best_color, 0, 255).astype(np.uint8)

    # ── Saturation boost — recovers vivid colours washed out by diffuse lighting
    hsv = cv2.cvtColor(colors.reshape(1, M, 3), cv2.COLOR_RGB2HSV).reshape(M, 3).astype(np.float32)
    hsv[:, 1] = np.clip(hsv[:, 1] * 1.8 + 25, 0, 255)   # multiply + floor lift
    colors = cv2.cvtColor(hsv.astype(np.uint8).reshape(1, M, 3), cv2.COLOR_HSV2RGB).reshape(M, 3)

    return colors


def _visual_hull_carving(
    silhouettes: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    orig_images: list[np.ndarray] | None = None,
    angles: np.ndarray | None = None,
    elevations: np.ndarray | None = None,
    grid_size: int   = GRID_SIZE,
    distance: float  = CAMERA_DISTANCE,
    elevation: float = CAMERA_ELEVATION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carve a dense voxel grid using normalised silhouettes and a vote threshold,
    enhanced with concavity-based veto carving.

    If orig_images (list of BGR arrays, same order as silhouettes) is provided,
    a second colour-sampling pass runs over occupied voxels only and returns
    per-voxel average RGB sampled from the original photographs.

    Returns:
        pts    (M, 3) float32 — world coordinates of occupied voxel centres
        colors (M, 3) uint8  — RGB colour per voxel (gray if no images given)
    """
    n_views = len(silhouettes)
    sz      = float(NORM_SIZE)
    # World ±1 at the orbit distance maps to the un-padded part of the window
    f_eff   = sz / (2.0 * (1.0 + 2.0 * NORM_PAD)) * distance
    cx = cy = sz / 2.0
    w  = h  = NORM_SIZE

    coords  = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    n_total = grid_size ** 3
    if angles is None:
        angles = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)
    if elevations is None:
        elevations = np.full(n_views, elevation)

    # Pre-compute silhouettes, concavity masks, camera matrices, and colour images
    view_data: list[tuple] = []
    windows = _crop_windows(silhouettes)
    for i, angle in enumerate(angles):
        orig_bgr = orig_images[i] if orig_images else None
        sil, cav, normed_rgb = _normalize_silhouette(
            silhouettes[i], windows[i], orig_bgr=orig_bgr,
            dilation_px=DILATION_PX, open_px=THIN_OPEN_PX,
        )
        R, t = _build_camera(angle, elevations[i], distance)
        view_data.append((sil, cav, R.astype(np.float32), t.astype(np.float32), normed_rgb))

    gs2 = grid_size * grid_size
    min_votes    = max(1, int(np.ceil(n_views * MIN_VOTE_FRAC)))
    allowed_miss = n_views - min_votes
    occupied     = np.zeros(n_total, dtype=bool)

    for chunk_start in range(0, n_total, _PROJ_CHUNK):
        chunk_end = min(chunk_start + _PROJ_CHUNK, n_total)

        # Indices of voxels still in the running; a voxel is dropped the moment
        # it can no longer reach min_votes (or trips the concavity veto), so
        # later views only project the survivors.
        alive  = np.arange(chunk_start, chunk_end, dtype=np.int32)
        misses = np.zeros(alive.size, dtype=np.int16)
        concav = np.zeros(alive.size, dtype=np.int16)

        for sil, cav, R, t, _normed_rgb in view_data:
            if not alive.size:
                break
            pts = np.stack([coords[alive // gs2],
                            coords[(alive // grid_size) % grid_size],
                            coords[alive % grid_size]], axis=1)
            pts_cam  = pts @ R.T + t
            z        = pts_cam[:, 2]
            in_front = z > 1e-4

            safe_z = np.where(in_front, z, 1.0)
            px_i = np.round(f_eff * pts_cam[:, 0] / safe_z + cx).astype(np.int32)
            py_i = np.round(f_eff * pts_cam[:, 1] / safe_z + cy).astype(np.int32)

            in_bounds = in_front & (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h)
            valid     = np.where(in_bounds)[0]

            in_sil = ~in_front                        # behind the camera: no evidence
            in_cav = np.zeros(alive.size, dtype=bool)
            if valid.size:
                in_sil[valid] = sil[py_i[valid], px_i[valid]] > 0
                in_cav[valid] = cav[py_i[valid], px_i[valid]] > 0

            misses += ~in_sil
            concav += in_cav
            keep = (misses <= allowed_miss) & (concav < CONCAVITY_VETO)
            alive, misses, concav = alive[keep], misses[keep], concav[keep]

        occupied[alive] = True

    occupied_3d = occupied.reshape(grid_size, grid_size, grid_size)
    del occupied

    # ── Keep only the largest connected component ────────────────────────────
    labeled, n_components = nd_label(occupied_3d)
    del occupied_3d
    if n_components == 0:
        empty_pts = np.empty((0, 3), dtype=np.float32)
        empty_clr = np.empty((0, 3), dtype=np.uint8)
        return empty_pts, empty_clr
    if n_components > 1:
        sizes         = np.bincount(labeled.ravel())[1:]
        largest_label = int(np.argmax(sizes)) + 1
        keep          = labeled == largest_label
        logger.info(
            "Connected components: %d found, kept largest (%d voxels), "
            "discarded %d voxels in satellite blobs",
            n_components,
            sizes[largest_label - 1],
            sizes.sum() - sizes[largest_label - 1],
        )
    else:
        keep = labeled > 0
    del labeled

    # Only the hull's surface is stored: the interior carries no colour
    # information and is re-filled by the voxelization stage.  This cuts the
    # serialised point cloud by roughly an order of magnitude.
    keep &= ~binary_erosion(keep, structure=generate_binary_structure(3, 1), border_value=0)

    # Reconstruct world coordinates for surface voxels
    occ_idx = np.where(keep.ravel())[0].astype(np.int32)
    del keep
    ix = occ_idx // gs2
    iy = (occ_idx // grid_size) % grid_size
    iz = occ_idx % grid_size
    occupied_pts = np.stack([coords[ix], coords[iy], coords[iz]], axis=1)

    # ── Colour sampling pass (only over surviving voxels — very fast) ────────
    colors = _sample_colors(occupied_pts, view_data, f_eff, cx, cy, w, h)

    return occupied_pts, colors


def view_poses(view_indices: list[int], n_uploaded: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Camera (azimuth, elevation) for each surviving photo, from its position in
    the upload order — so a photo rejected by segmentation leaves a gap instead
    of shifting every later view onto the wrong bearing.

    Captures of TWO_RING_MIN or more photos (even count) are read as two rings,
    as the UI recommends: the first half level with the object, the second
    half from above at the same bearings.
    """
    two_rings = n_uploaded >= TWO_RING_MIN and n_uploaded % 2 == 0
    per_ring  = n_uploaded // 2 if two_rings else n_uploaded
    idx       = np.asarray(view_indices)
    angles    = 2.0 * np.pi * (idx % per_ring) / per_ring
    elevations = np.where(idx >= per_ring, HIGH_ELEVATION, CAMERA_ELEVATION) if two_rings \
        else np.full(len(idx), CAMERA_ELEVATION)
    return angles, elevations


def carve_views(
    view_indices: list[int],
    n_uploaded: int,
    silhouettes: list[np.ndarray],
    orig_images: list[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Carve the visual hull from the views that survived segmentation."""
    angles, elevations = view_poses(view_indices, n_uploaded)
    image_sizes = [(s.shape[1], s.shape[0]) for s in silhouettes]
    return _visual_hull_carving(silhouettes, image_sizes, orig_images,
                                angles=angles, elevations=elevations)


def to_point_list(pts: np.ndarray, colors: np.ndarray) -> list[dict]:
    """Serialise the carved hull surface as {x, y, z, r, g, b} dicts (Y up)."""
    return [
        {
            "x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "z": round(float(p[2]), 4),
            "r": int(c[0]), "g": int(c[1]), "b": int(c[2]),
        }
        for p, c in zip(pts, colors)
    ]


# ── Async pipeline stage ──────────────────────────────────────────────────────

async def run_reconstruction(run_id: str) -> dict:
    db = get_db()
    fs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("segmented",):
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run['status']}', expected 'segmented'",
        )

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {
            "status": "reconstructing",
            "reconstruction_started_at": datetime.now(timezone.utc),
        }},
    )

    try:
        seg_images = run.get("segmented_images", [])
        if len(seg_images) < 2:
            raise ValueError("Need at least 2 segmented images for visual hull reconstruction")

        loop = asyncio.get_event_loop()

        silhouettes : list[np.ndarray]        = []
        orig_images : list[np.ndarray | None] = []
        view_indices: list[int]               = []

        for pos, entry in enumerate(seg_images):
            # Load segmented mask
            seg_stream = await fs.open_download_stream(ObjectId(entry["segmented_file_id"]))
            seg_data   = await seg_stream.read()
            sil, _     = await loop.run_in_executor(None, _extract_silhouette, seg_data)
            silhouettes.append(sil)
            view_indices.append(entry.get("view_index", pos))

            # Load original colour image for colour sampling
            orig_stream = await fs.open_download_stream(ObjectId(entry["original_file_id"]))
            orig_data   = await orig_stream.read()
            buf         = np.frombuffer(orig_data, dtype=np.uint8)
            bgr         = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                # cv2 can't decode HEIC/HEIF — fall back to Pillow
                try:
                    from PIL import Image as _PILImage
                    pil_img = _PILImage.open(io.BytesIO(orig_data)).convert("RGB")
                    bgr     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    bgr = None
            if bgr is None:
                logger.warning("Could not decode original image for %s — colours will be gray", entry.get("filename", "?"))
            orig_images.append(bgr)

        n_uploaded = len(run.get("images", [])) or len(seg_images)
        logger.info("Starting visual hull carving for run %s (%d/%d views)", run_id, len(silhouettes), n_uploaded)

        pts, colors = await loop.run_in_executor(
            None, carve_views, view_indices, n_uploaded, silhouettes, orig_images,
        )

        if pts.shape[0] == 0:
            raise ValueError(
                "Visual hull is empty — silhouettes may not overlap. "
                "Try more images from additional angles."
            )

        logger.info("Visual hull: %d occupied voxels for run %s", pts.shape[0], run_id)

        point_list = to_point_list(pts, colors)
        pts_bytes = json.dumps(point_list).encode()

        cloud_id = await fs.upload_from_stream(
            "point_cloud.json",
            io.BytesIO(pts_bytes),
            metadata={
                "run_id":       run_id,
                "content_type": "application/json",
                "stage":        "reconstruction",
            },
        )

        meta = {
            "point_cloud_file_id": str(cloud_id),
            "point_count":         pts.shape[0],
            "method":              "visual_hull",
            "grid_size":           GRID_SIZE,
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status":                       "reconstructed",
                "reconstruction":               meta,
                "reconstruction_completed_at":  datetime.now(timezone.utc),
            }},
        )

        return {"run_id": run_id, "status": "reconstructed", **meta}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Reconstruction failed for run %s", run_id)
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
