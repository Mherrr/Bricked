"""
Score the Bricked pipeline against rendered ground truth.

Runs the backend's own stage functions (segmentation → visual hull →
voxelization → LEGO packing) on every scene produced by render_turntable.py and
reports, per object and on average:

  * segmentation: pass rate, mask IoU vs ground-truth mask, YOLO confidence
  * reconstruction: 3-D IoU of the carved hull vs the ground-truth solid
    (both normalised to their bounding box, compared on a 64^3 grid)
  * LEGO: voxel / brick counts, studs per brick, palette size,
    colour error (CIE76 ΔE between voxel colour and assigned brick colour)
  * wall-clock time per stage

    python evaluate.py /path/to/scenes --json results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services import lego_service, reconstruction_service, segmentation_service, voxel_service  # noqa: E402

RES = 64


def _occupancy(pts: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Rasterise points into RES^3 after mapping their bounding box [lo, hi] into [-1, 1]^3."""
    n = (pts - (lo + hi) / 2.0) / (np.max(hi - lo) / 2.0 + 1e-9)
    idx = np.clip(np.floor((n + 1.0) / 2.0 * RES).astype(int), 0, RES - 1)
    occ = np.zeros((RES,) * 3, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def _cells_to_points(cells: np.ndarray, sub: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample each unit cell [i, i+1)^3 on a sub^3 lattice; return samples and cell-extent bbox."""
    o = (np.arange(sub) + 0.5) / sub
    off = np.stack(np.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
    pts = (cells[:, None, :] + off[None]).reshape(-1, 3)
    return pts, cells.min(0), cells.max(0) + 1.0


def iou3d(pred: np.ndarray, gt_occ: np.ndarray, cells: bool = False, y_scale: float = 1.0) -> float:
    """IoU of a prediction and the GT solid, both normalised to their own bounding box.

    pred is either dense points (the 256^3 hull) or integer cell indices
    (coarse voxels, cells=True) which are sampled so each cell is filled.
    """
    if cells:
        p, plo, phi = _cells_to_points(pred.astype(np.float64), sub=max(1, int(np.ceil(RES / (np.ptp(pred, 0).max() + 1))) * 2))
        stretch = np.array([1.0, y_scale, 1.0])
        p, plo, phi = p * stretch, plo * stretch, phi * stretch
    else:
        p, plo, phi = pred, pred.min(0), pred.max(0)
    g, glo, ghi = _cells_to_points(np.argwhere(gt_occ).astype(np.float64), sub=1)
    # The hull is stored as a surface shell; score the solid it encloses
    a, b = binary_fill_holes(_occupancy(p, plo, phi)), _occupancy(g, glo, ghi)
    return float((a & b).sum() / max((a | b).sum(), 1))


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    return float((p & g).sum() / max((p | g).sum(), 1))


def visible_cells(cells: set) -> set:
    """Cells with at least one open face (kept local so older pipelines can be scored)."""
    steps = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    return {(x, y, z) for (x, y, z) in cells
            if any((x + dx, y + dy, z + dz) not in cells for dx, dy, dz in steps)}


def colour_error(bricks: list[dict], voxels: list[dict]) -> float:
    """Mean CIE76 ΔE between each visible voxel's colour and the brick colour covering it."""
    if not voxels or "r" not in voxels[0]:
        return float("nan")
    cell = {}
    for b in bricks:
        rgb = tuple(int(b["color"][i:i + 2], 16) for i in (1, 3, 5))
        for dx in range(b["width"]):
            for dz in range(b["depth"]):
                cell[(b["x"] + dx, b["y"], b["z"] + dz)] = rgb
    # Only visible voxels count — interior colour is never seen
    shown = visible_cells({(v["x"], v["y"], v["z"]) for v in voxels})
    vs = [v for v in voxels if (v["x"], v["y"], v["z"]) in shown]
    src = np.array([[v["r"], v["g"], v["b"]] for v in vs], np.uint8)
    dst = np.array([cell[(v["x"], v["y"], v["z"])] for v in vs], np.uint8)
    to_lab = lambda a: cv2.cvtColor(a.reshape(-1, 1, 3).astype(np.float32) / 255.0,
                                    cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return float(np.linalg.norm(to_lab(src) - to_lab(dst), axis=1).mean())


def structure(bricks: list[dict]) -> tuple[float, float]:
    """
    (bond ratio, largest connected share).

    bond ratio  — of the bricks resting on something, the share that span two
                  or more bricks below (a running bond, not stacked columns).
    connected   — share of bricks in the largest stud-connected assembly.
    """
    owner = {}
    for i, b in enumerate(bricks):
        for dx in range(b["width"]):
            for dz in range(b["depth"]):
                owner[(b["x"] + dx, b["y"], b["z"] + dz)] = i
    parent = list(range(len(bricks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    supported = bonded = 0
    for i, b in enumerate(bricks):
        below = {owner.get((b["x"] + dx, b["y"] - 1, b["z"] + dz))
                 for dx in range(b["width"]) for dz in range(b["depth"])} - {None}
        for j in below:
            parent[find(i)] = find(j)
        if below:
            supported += 1
            bonded += len(below) >= 2
    roots = np.bincount([find(i) for i in range(len(bricks))]) if bricks else np.array([0])
    return bonded / max(supported, 1), roots.max() / max(len(bricks), 1)


def run_scene(scene: Path, use_gt_masks: bool = False) -> dict:
    img_paths = sorted((scene / "images").glob("*.jpg"))
    gt = np.load(scene / "gt.npz")["occupancy"]
    r: dict = {"name": scene.name, "views": len(img_paths)}

    # ── Segmentation ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    kept, seg_ious, confs, skipped = [], [], [], []
    for i, p in enumerate(img_paths):
        data = p.read_bytes()
        gt_mask = cv2.imread(str(scene / "masks" / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        try:
            png, meta = segmentation_service._segment_image(data)
        except ValueError as exc:
            skipped.append({"view": i, "reason": str(exc)[:80]})
            continue
        sil, _ = reconstruction_service._extract_silhouette(png)
        seg_ious.append(mask_iou(sil, gt_mask))
        confs.append(meta["confidence"])
        if use_gt_masks:
            sil = (gt_mask > 0).astype(np.uint8)
        kept.append((i, sil, cv2.imread(str(p), cv2.IMREAD_COLOR)))
    r["t_segment"] = time.perf_counter() - t0
    r["seg_pass"] = len(kept) / len(img_paths)
    r["seg_iou"] = float(np.mean(seg_ious)) if seg_ious else 0.0
    r["seg_conf"] = float(np.mean(confs)) if confs else 0.0
    r["skipped"] = skipped
    if len(kept) < 2:
        r["failed"] = "fewer than 2 views passed segmentation"
        return r

    # ── Reconstruction ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    pts, colors = reconstruction_service.carve_views(
        view_indices=[k[0] for k in kept],
        n_uploaded=len(img_paths),
        silhouettes=[k[1] for k in kept],
        orig_images=[k[2] for k in kept],
    )
    r["t_reconstruct"] = time.perf_counter() - t0
    if len(pts) == 0:
        r["failed"] = "empty hull"
        return r
    point_list = reconstruction_service.to_point_list(pts, colors)
    r["points"] = len(point_list)
    r["iou3d"] = iou3d(np.array([[p["x"], p["y"], p["z"]] for p in point_list]), gt)

    # ── Voxelization ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    voxels = voxel_service._build_voxel_grid(point_list)
    r["t_voxelize"] = time.perf_counter() - t0
    r["voxels"] = len(voxels)
    # Voxels are one brick tall (1.2 stud pitches): restore true proportions before scoring
    vox_idx = np.array([[v["x"], v["y"], v["z"]] for v in voxels], float)
    r["iou3d_voxels"] = iou3d(vox_idx, gt, cells=True, y_scale=getattr(voxel_service, "BRICK_ASPECT", 1.0)) if voxels else 0.0

    # ── LEGO ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    bricks = lego_service._pack_bricks(voxels)
    r["t_lego"] = time.perf_counter() - t0
    r["bricks"] = len(bricks)
    r["studs_per_brick"] = len(voxels) / max(len(bricks), 1)
    r["colors"] = len({b["color_name"] for b in bricks})
    r["bond_ratio"], r["connected"] = structure(bricks)
    r["delta_e"] = colour_error(bricks, voxels)
    r["t_total"] = r["t_segment"] + r["t_reconstruct"] + r["t_voxelize"] + r["t_lego"]
    return r


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if "failed" not in r]
    m = lambda k: float(np.nanmean([r[k] for r in ok])) if ok else float("nan")
    return {
        "objects": len(rows),
        "completed": len(ok),
        "seg_pass_rate": float(np.mean([r["seg_pass"] for r in rows])),
        "seg_mask_iou": float(np.mean([r["seg_iou"] for r in rows if r["seg_iou"] > 0])),
        "seg_confidence": float(np.mean([r["seg_conf"] for r in rows if r["seg_conf"] > 0])),
        "iou3d_hull": m("iou3d"),
        "iou3d_hull_median": float(np.median([r["iou3d"] for r in ok])) if ok else float("nan"),
        "iou3d_voxels": m("iou3d_voxels"),
        "voxels": m("voxels"),
        "bricks": m("bricks"),
        "studs_per_brick": m("studs_per_brick"),
        "colors": m("colors"),
        "bond_ratio": m("bond_ratio"),
        "connected": m("connected"),
        "delta_e": m("delta_e"),
        "t_segment": m("t_segment"),
        "t_reconstruct": m("t_reconstruct"),
        "t_voxelize": m("t_voxelize"),
        "t_lego": m("t_lego"),
        "t_total": m("t_total"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--gt-masks", action="store_true",
                    help="carve from ground-truth masks (isolates reconstruction from segmentation)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    segmentation_service._load_model()                # exclude one-off model load from timings
    rows = []
    for scene in sorted(p for p in args.scenes.iterdir() if (p / "gt.npz").exists()):
        if args.only and scene.name not in args.only:
            continue
        r = run_scene(scene, use_gt_masks=args.gt_masks)
        rows.append(r)
        print(f"{r['name']:<24} pass={r['seg_pass']:.2f} segIoU={r['seg_iou']:.3f} "
              f"IoU3d={r.get('iou3d', 0):.3f} vox={r.get('voxels', 0):>5} bricks={r.get('bricks', 0):>4} "
              f"colors={r.get('colors', 0)} bond={r.get('bond_ratio', 0):.2f} dE={r.get('delta_e', float('nan')):.1f} "
              f"t={r.get('t_total', 0):.1f}s {r.get('failed', '')}", flush=True)

    s = summarise(rows)
    print(json.dumps(s, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": s, "objects": rows}, indent=2))


if __name__ == "__main__":
    main()
