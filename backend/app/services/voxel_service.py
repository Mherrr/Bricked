"""
Voxelization stage using Open3D + scipy.

Pipeline:
  1. Load reconstructed point cloud from GridFS
  2. Convert to stud space (brick-height Y) and pick the voxel pitch
  3. Open3D VoxelGrid in brick proportions (1 stud × 1 brick height), sized so
     the longest side is TARGET_STUDS; the hull surface is then filled solid
  4. Gaussian smoothing of the 3-D occupancy grid — replaces jagged/complex
     surface detail with smooth, simplified geometry:
       · 1-voxel protrusions fall below the threshold and are removed
       · shallow concavities are filled (surrounded voxels stay above threshold)
       · sharp corners round into smooth curves
  5. Morphological opening (erode → dilate) — removes any remaining silhouette
     fins or thin wings that survive the Gaussian pass
  6. Largest-connected-component filter — drops pieces detached by the opening
  7. Store cleaned voxel list to GridFS
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
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label as nd_label,
)

from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

# ── Simplification parameters ─────────────────────────────────────────────────
#
# TARGET_STUDS controls the coarseness of the output grid: the object's longest
# dimension becomes this many studs (or the equivalent in brick heights), so
# model size no longer depends on how the photos happened to be framed.
# Cells are one stud wide and one brick tall — a LEGO brick is 1.2× taller
# than its stud pitch — so the built model keeps the object's proportions.
#
# GAUSS_SIGMA / GAUSS_THRESH set how aggressively the occupancy grid is
# smoothed before re-binarising: a larger sigma removes 1-voxel bumps and fills
# shallow concavities, at the cost of rounding real corners.  The benchmark
# (benchmark/evaluate.py) found shape accuracy flat for sigma 0.2–0.5, so a
# light touch is used.  OPEN_ITERS > 0 adds a morphological opening to strip
# thin fins; it is off because thin parts (legs, handles) are usually real.

TARGET_STUDS = 28     # longest side of the model, in studs
BRICK_ASPECT = 1.2    # brick height / stud pitch
GAUSS_SIGMA  = 0.2    # smoothing radius in voxels
GAUSS_THRESH = 0.45   # re-binarisation threshold after smoothing
OPEN_ITERS   = 0      # morphological opening iterations (fin removal)


def _build_voxel_grid(point_list: list[dict]) -> list[dict]:
    """
    Convert a point cloud (list of {x,y,z[,r,g,b]} dicts in [-1,1]³) to a
    simplified, LEGO-ready voxel grid.  Returns integer grid-index dicts
    {x, y, z, r, g, b}.
    """
    import open3d as o3d

    if not point_list:
        return []

    pts      = np.array([[p["x"], p["y"], p["z"]] for p in point_list], dtype=np.float64)
    has_color = "r" in point_list[0]

    # Work in "stud space": squash Y so a cubic voxel becomes one brick tall,
    # and pick the pitch that makes the longest side TARGET_STUDS cells.
    pts[:, 1] /= BRICK_ASPECT
    voxel_size = float(np.ptp(pts, axis=0).max()) / TARGET_STUDS or 1.0

    # ── 1. Point cloud (no outlier removal: the reconstruction stage already
    #        keeps only the hull's largest connected component, and statistical
    #        outlier removal was measured to strip real thin geometry)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if has_color:
        rgb = np.array([[p["r"], p["g"], p["b"]] for p in point_list], dtype=np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    # ── 2. Coarse voxelisation ────────────────────────────────────────────────
    vg         = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    raw_voxels = vg.get_voxels()
    if not raw_voxels:
        return []

    # ── 3. Build 3-D occupancy grid + color map ───────────────────────────────
    indices = np.array([v.grid_index for v in raw_voxels], dtype=np.int32)
    # +2 padding so the Gaussian and erosion kernels don't clip grid edges
    shape   = tuple(indices.max(axis=0) + 2)
    grid    = np.zeros(shape, dtype=np.float32)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    n_pre   = int((grid > 0).sum())

    # color_map: grid-index tuple → [R, G, B] uint8
    color_map: dict[tuple, np.ndarray] = {}
    if has_color:
        for v in raw_voxels:
            ix, iy, iz = v.grid_index
            color_map[(ix, iy, iz)] = np.array(
                [int(round(c * 255)) for c in v.color], dtype=np.uint8
            )

    # ── 3b. Saturation boost — Open3D averages colors across all points in a
    #        voxel cell, which pulls mixed-surface cells toward gray.  Boost
    #        saturation to restore vivid colors before downstream quantization.
    if has_color and color_map:
        keys   = list(color_map.keys())
        rgb_arr = np.array([color_map[k] for k in keys], dtype=np.uint8).reshape(-1, 1, 3)
        hsv     = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, 0, 1] = np.clip(hsv[:, 0, 1] * 1.5, 0, 255)
        boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(-1, 3)
        for i, k in enumerate(keys):
            color_map[k] = boosted[i]

    # ── 3c. Solidify — the reconstruction stage stores only the hull surface,
    #        so fill the enclosed interior before simplifying the shape.
    grid = binary_fill_holes(grid > 0).astype(np.float32)

    # ── 4. Gaussian smoothing — shape simplification ──────────────────────────
    smoothed = gaussian_filter(grid, sigma=GAUSS_SIGMA)
    grid_bin = smoothed >= GAUSS_THRESH

    logger.debug(
        "Gaussian (σ=%.1f, thresh=%.2f): %d → %d voxels",
        GAUSS_SIGMA, GAUSS_THRESH, n_pre, int(grid_bin.sum()),
    )

    # ── 5. Morphological opening — remove remaining thin fins ─────────────────
    # (scipy treats iterations=0 as "repeat until nothing changes", which
    #  erodes the whole grid away — so skip the opening entirely when disabled)
    if OPEN_ITERS > 0:
        struct  = generate_binary_structure(3, 1)   # 6-connected face kernel
        eroded  = binary_erosion(grid_bin, structure=struct, iterations=OPEN_ITERS, border_value=0)
        cleaned = binary_dilation(eroded,  structure=struct, iterations=OPEN_ITERS)
    else:
        cleaned = grid_bin

    # ── 6. Largest connected component ────────────────────────────────────────
    labeled, n_comp = nd_label(cleaned)
    if n_comp == 0:
        # Simplification removed everything (object too thin) — fall back
        logger.warning("Simplified grid is empty; falling back to raw grid")
        labeled, n_comp = nd_label(grid_bin if grid_bin.any() else (grid > 0))
    if n_comp == 0:
        return []

    sizes         = np.bincount(labeled.ravel())[1:]
    largest_label = int(np.argmax(sizes)) + 1
    final_grid    = labeled == largest_label
    n_post        = int(final_grid.sum())

    logger.info(
        "Voxelization: %d raw → %d simplified (%.1f%% reduction | %d extra component%s dropped)",
        n_pre, n_post,
        100.0 * (n_pre - n_post) / max(n_pre, 1),
        n_comp - 1, "s" if n_comp - 1 != 1 else "",
    )

    xi, yi, zi = np.where(final_grid)

    if not has_color:
        return [{"x": int(x), "y": int(y), "z": int(z)} for x, y, z in zip(xi, yi, zi)]

    # ── 7. Assign colors to final voxels ─────────────────────────────────────
    # Every voxel takes the colour of the nearest cell that received one from
    # the point cloud (interior cells filled in step 3c, and any cell the
    # smoothing added, inherit from the closest surface).
    has_rgb = np.ones(final_grid.shape, dtype=bool)
    for key in color_map:
        if all(k < n for k, n in zip(key, final_grid.shape)):
            has_rgb[key] = False                      # EDT measures distance to zeros
    _, (ni, nj, nk) = distance_transform_edt(has_rgb, return_indices=True)
    _GRAY = np.array([128, 128, 128], dtype=np.uint8)

    result = []
    for x, y, z in zip(xi, yi, zi):
        src = (int(ni[x, y, z]), int(nj[x, y, z]), int(nk[x, y, z]))
        rgb = color_map.get(src, _GRAY)
        result.append({
            "x": int(x), "y": int(y), "z": int(z),
            "r": int(rgb[0]), "g": int(rgb[1]), "b": int(rgb[2]),
        })
    return result


async def run_voxelization(run_id: str) -> dict:
    db     = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("reconstructed",):
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run['status']}', expected 'reconstructed'",
        )

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "voxelizing", "voxelization_started_at": datetime.now(timezone.utc)}},
    )

    try:
        recon      = run.get("reconstruction")
        stream     = await gridfs.open_download_stream(ObjectId(recon["point_cloud_file_id"]))
        point_list = json.loads(await stream.read())

        loop   = asyncio.get_event_loop()
        voxels = await loop.run_in_executor(None, _build_voxel_grid, point_list)

        if not voxels:
            raise ValueError("Voxel grid is empty after simplification — point cloud may be too sparse")

        logger.info("Voxelized run %s: %d voxels (%d studs across)", run_id, len(voxels), TARGET_STUDS)

        voxel_bytes   = json.dumps(voxels).encode()
        voxel_file_id = await gridfs.upload_from_stream(
            "voxels.json",
            io.BytesIO(voxel_bytes),
            metadata={"run_id": run_id, "content_type": "application/json", "stage": "voxelization"},
        )

        voxel_meta = {
            "voxel_file_id": str(voxel_file_id),
            "voxel_count":   len(voxels),
            "voxel_size":    round(1.0 / TARGET_STUDS, 4),   # fraction of the longest side
            "stud_span":     TARGET_STUDS,
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status":                    "voxelized",
                "voxelization":              voxel_meta,
                "voxelization_completed_at": datetime.now(timezone.utc),
            }},
        )
        return {"run_id": run_id, "status": "voxelized", **voxel_meta}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Voxelization failed for run %s", run_id)
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
