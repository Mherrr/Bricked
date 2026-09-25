"""
LEGO conversion stage.
Reads the voxel grid, merges voxels into standard brick types, assigns colors,
generates a parts list, and stores the brick layout as JSON for the Three.js
renderer.

Brick types supported: 1x1, 1x2, 1x3, 1x4, 2x2, 2x3, 2x4 (either orientation).
Colors are quantized to a palette of real LEGO colors via nearest-neighbour
search in CIE-LAB space.
"""
import cv2
import json
import io
import numpy as np
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs

# Standard LEGO color palette (name → (hex, R, G, B)); hex values follow Rebrickable
LEGO_PALETTE: dict[str, tuple[str, int, int, int]] = {
    "Bright Red":           ("#C91A09", 201,  26,   9),
    "Dark Red":             ("#720E0F", 114,  14,  15),
    "Bright Blue":          ("#0055BF",   0,  85, 191),
    "Medium Blue":          ("#5A93DB",  90, 147, 219),
    "Dark Blue":            ("#0A3463",  10,  52,  99),
    "Bright Light Blue":    ("#9FC3E9", 159, 195, 233),
    "Sand Blue":            ("#6074A1",  96, 116, 161),
    "Medium Azure":         ("#36AEBF",  54, 174, 191),
    "Dark Azure":           ("#078BC9",   7, 139, 201),
    "Light Aqua":           ("#B3D7D1", 179, 215, 209),
    "Bright Yellow":        ("#F2CD37", 242, 205,  55),
    "Bright Light Orange":  ("#F8BB3D", 248, 187,  61),
    "Bright Orange":        ("#FE8A18", 254, 138,  24),
    "Dark Orange":          ("#A95500", 169,  85,   0),
    "Bright Green":         ("#4B9F4A",  75, 159,  74),
    "Dark Green":           ("#184632",  24,  70,  50),
    "Lime":                 ("#BBE90B", 187, 233,  11),
    "Yellowish Green":      ("#DFEEA5", 223, 238, 165),
    "Olive Green":          ("#9B9A5A", 155, 154,  90),
    "Sand Green":           ("#A0BCAC", 160, 188, 172),
    "White":                ("#FFFFFF", 255, 255, 255),
    "Black":                ("#05131D",   5,  19,  29),
    "Medium Stone Gray":    ("#A0A5A9", 160, 165, 169),
    "Dark Stone Gray":      ("#6C6E68", 108, 110, 104),
    "Reddish Brown":        ("#582A12",  88,  42,  18),
    "Dark Brown":           ("#352100",  53,  33,   0),
    "Nougat":               ("#D09168", 208, 145, 104),
    "Medium Nougat":        ("#AA7D55", 170, 125,  85),
    "Light Nougat":         ("#F6D7B3", 246, 215, 179),
    "Tan":                  ("#E4CD9E", 228, 205, 158),
    "Dark Tan":             ("#958A73", 149, 138, 115),
    "Coral":                ("#FF698F", 255, 105, 143),
    "Bright Pink":          ("#E4ADC8", 228, 173, 200),
    "Magenta":              ("#923978", 146,  57, 120),
    "Bright Purple":        ("#81007B", 129,   0, 123),
    "Medium Lavender":      ("#AC78BA", 172, 120, 186),

    # ── Neutral ramp ────────────────────────────────────────────────────────
    # The greys carry every uncoloured object, and the ramp had holes of 83,
    # 54 and 104 units of lightness between White, Medium Stone Gray, Dark
    # Stone Gray and Black — so distinct greys on a photo collapsed onto one
    # brick.  These are the real LEGO greys that land inside those holes.
    # Flat Silver, Pearl Dark Gray and Pearl Gold have a pearlescent finish
    # and are less widely stocked in every brick shape than the solid colours.
    "Very Light Bluish Gray": ("#E6E3E0", 230, 227, 224),
    "Flat Silver":            ("#898788", 137, 135, 136),
    "Pearl Dark Gray":        ("#575857",  87,  88,  87),

    # ── Additional solid colours ────────────────────────────────────────────
    "Green":                ("#237841",  35, 120,  65),
    "Medium Green":         ("#73DCA1", 115, 220, 161),
    "Dark Turquoise":       ("#008F9B",   0, 143, 155),
    "Dark Blue Violet":     ("#2032B0",  32,  50, 176),
    "Dark Purple":          ("#3F3691",  63,  54, 145),
    "Lavender":             ("#E1D5ED", 225, 213, 237),
    "Dark Pink":            ("#C870A0", 200, 112, 160),
    "Light Pink":           ("#FC97AC", 252, 151, 172),
    "Sand Red":             ("#D67572", 214, 117, 114),
    "Brown":                ("#583927",  88,  57,  39),
    "Pearl Gold":           ("#AA7F2E", 170, 127,  46),
    "Medium Orange":        ("#FFA70B", 255, 167,  11),
    "Bright Light Yellow":  ("#FFF03A", 255, 240,  58),
}

# Pre-compute full palette in CIE-LAB for perceptually-uniform nearest-color lookup.
_PALETTE_NAMES = list(LEGO_PALETTE.keys())

def _rgb_to_lab(r: int, g: int, b: int) -> np.ndarray:
    """Convert a single sRGB pixel to OpenCV's uint8 LAB encoding."""
    px = np.array([[[r, g, b]]], dtype=np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

def _rgb_array_to_lab(arr: np.ndarray) -> np.ndarray:
    """Batch convert (N, 3) uint8 RGB array to (N, 3) float32 LAB."""
    return cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

_PALETTE_LAB = np.stack([
    _rgb_to_lab(r, g, b) for _, r, g, b in LEGO_PALETTE.values()
])  # (N, 3)

# Supported brick footprints (width x depth in stud units)
BRICK_TYPES = [(2, 4), (2, 3), (2, 2), (1, 4), (1, 3), (1, 2), (1, 1)]

# Score bonus per distinct brick a candidate spans in the layer below: a brick
# bridging two bricks beats one up to three studs larger that sits on just one.
# (Benchmarked: raises the bonded share from 0.64 to 0.68 at no brick-count cost.)
BOND_WEIGHT = 3.0

# Footprints tried per layer, largest first, in both orientations.  Even layers
# prefer bricks running along Z, odd layers along X, so the seams of one layer
# are crossed by the bricks above it (a running bond) instead of stacking into
# vertical columns that would fall apart.
def _layer_footprints(y: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for (w, d) in BRICK_TYPES:
        pair = [(w, d), (d, w)] if y % 2 == 0 else [(d, w), (w, d)]
        for fp in pair:
            if fp not in out:
                out.append(fp)
    return out

# Number of k-means clusters used to derive the dominant color palette.
# Raise to allow more colors; lower to be stricter about phantom-color suppression.
_N_COLOR_CLUSTERS = 6

# LAB chroma below which a colour counts as neutral (grey / white / black).
# The palette's neutral ramp is sparse — White, Medium Stone Gray, Dark Stone
# Gray — while a dozen low-chroma tinted colours (Dark Tan, Sand Green, Tan,
# Light Nougat, Sand Blue) sit in the lightness gaps between them.  Plain
# Euclidean LAB distance therefore matches a perfectly neutral grey to Dark Tan,
# because closeness in lightness outweighs being off the grey axis.  Neutral
# samples are matched only against neutral bricks so grey stays grey.
NEUTRAL_CHROMA = 12.0

# A sample counts as neutral a little further off the axis than a brick does.
# After white balance a grey surface still carries a few units of chroma from
# sensor noise and k-means centroid scatter; without the wider query threshold
# those land on Dark Tan (chroma 14) or Light Aqua (13), which are exactly the
# colours the guard exists to avoid.  Bricks stay judged at the tighter value,
# so those two never become targets themselves.
NEUTRAL_QUERY_CHROMA = 18.0

_PALETTE_CHROMA = np.hypot(_PALETTE_LAB[:, 1] - 128.0, _PALETTE_LAB[:, 2] - 128.0)
_PALETTE_IS_NEUTRAL = _PALETTE_CHROMA < NEUTRAL_CHROMA


def _chroma(lab: np.ndarray) -> float:
    """Distance of a LAB colour from the neutral (grey) axis."""
    return float(np.hypot(lab[1] - 128.0, lab[2] - 128.0))


def _nearest_index(query: np.ndarray, lab_array: np.ndarray) -> int:
    """
    Index of the closest colour in lab_array, keeping neutrals neutral.

    When the query sits on the grey axis and the palette offers any neutral
    entry, only neutral entries are considered; a slightly wrong grey level
    reads far better than a grey turned tan.
    """
    diffs = ((lab_array - query) ** 2).sum(axis=1)
    if _chroma(query) < NEUTRAL_QUERY_CHROMA:
        neutral = np.hypot(lab_array[:, 1] - 128.0, lab_array[:, 2] - 128.0) < NEUTRAL_CHROMA
        if neutral.any():
            diffs = np.where(neutral, diffs, np.inf)
    return int(np.argmin(diffs))


def _quantize_against(r: int, g: int, b: int,
                      names: list[str], lab_array: np.ndarray) -> tuple[str, str]:
    """Nearest-LEGO-color lookup against an arbitrary (names, lab_array) palette."""
    idx  = _nearest_index(_rgb_to_lab(r, g, b), lab_array)
    name = names[idx]
    return name, LEGO_PALETTE[name][0]


def _dominant_palette(voxels: list[dict]) -> tuple[list[str], np.ndarray]:
    """
    K-means palette derivation in CIE-LAB space:
      1. Convert all voxel colors to LAB for perceptually-uniform clustering.
      2. Run k-means to find _N_COLOR_CLUSTERS perceptual color groups.
      3. Map each cluster centroid to the nearest LEGO palette color.

    Clustering on centroids rather than individual voxels means noise voxels
    with stray colors (e.g. a handful of reddish points on a brown surface)
    get absorbed into the dominant cluster instead of contributing a phantom
    LEGO color.
    """
    from scipy.cluster.vq import kmeans as _kmeans

    rgb_arr = np.array([[v["r"], v["g"], v["b"]] for v in voxels], dtype=np.uint8)
    lab_arr = _rgb_array_to_lab(rgb_arr)  # (N, 3) float32

    k = min(_N_COLOR_CLUSTERS, len(lab_arr))
    try:
        centroids, _ = _kmeans(lab_arr.astype(np.float64), k, iter=20)
    except Exception:
        centroids = lab_arr[:k].astype(np.float64)

    used_names: list[str] = []
    for c in centroids:
        idx   = _nearest_index(c.astype(np.float32), _PALETTE_LAB)
        name  = _PALETTE_NAMES[idx]
        if name not in used_names:
            used_names.append(name)

    if not used_names:
        used_names = ["Medium Stone Gray"]

    restricted_lab = np.stack([_rgb_to_lab(*LEGO_PALETTE[n][1:]) for n in used_names])
    return used_names, restricted_lab


def _visible_cells(cells: set[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    """Cells with at least one open face — the only ones anyone will see."""
    steps = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    return {
        (x, y, z) for (x, y, z) in cells
        if any((x + dx, y + dy, z + dz) not in cells for dx, dy, dz in steps)
    }


def _pack_bricks(voxels: list[dict]) -> list[dict]:
    """
    Color-aware greedy brick packing with dominant-palette filtering:
      1. Derive the dominant LEGO palette from the colours of the *visible*
         voxels (k-means in CIE-LAB, see _dominant_palette).
      2. Pack layer by layer.  A footprint is accepted when every visible cell
         under it shares one quantized colour; hidden interior cells are
         wildcards, so the core of the model packs into large bricks instead
         of being fragmented by colour noise nobody can see.
      3. Candidate footprints are scored by size, plus a bonus for each brick
         they bridge in the layer below; layers alternate orientation so seams
         are staggered (a running bond).
    """
    _default = ("Medium Stone Gray", LEGO_PALETTE["Medium Stone Gray"][0])

    cells = {(v["x"], v["y"], v["z"]) for v in voxels}
    visible = _visible_cells(cells)
    has_color = bool(voxels) and "r" in voxels[0]
    qcolor: dict[tuple, tuple[str, str]] = {}      # visible cells only

    if has_color:
        shown = [v for v in voxels if (v["x"], v["y"], v["z"]) in visible] or voxels
        dom_names, dom_lab = _dominant_palette(shown)
        for v in shown:
            name, hex_col = _quantize_against(v["r"], v["g"], v["b"], dom_names, dom_lab)
            qcolor[(v["x"], v["y"], v["z"])] = (name, hex_col)
    else:
        for c in visible:
            qcolor[c] = _default

    # Hidden bricks take the model's most common colour (cheapest to source)
    counts: dict[tuple[str, str], int] = {}
    for c in qcolor.values():
        counts[c] = counts.get(c, 0) + 1
    interior_color = max(counts, key=counts.get) if counts else _default

    by_layer: dict[int, set[tuple[int, int]]] = {}
    for v in voxels:
        by_layer.setdefault(v["y"], set()).add((v["x"], v["z"]))

    bricks = []
    owner: dict[tuple[int, int, int], int] = {}      # cell → index of the brick covering it
    for y in sorted(by_layer):
        remaining = set(by_layer[y])
        footprints = _layer_footprints(y)
        # Alternate the scan direction too, so brick edges shift between layers
        order = sorted(remaining) if y % 2 == 0 else sorted(remaining, key=lambda c: (c[1], c[0]))

        for (x, z) in order:
            if (x, z) not in remaining:
                continue
            best, best_score = None, -1.0
            for (w, d) in footprints:
                footprint = [(x + dx, z + dz) for dx in range(w) for dz in range(d)]
                if not all(c in remaining for c in footprint):
                    continue
                shown_colors = {qcolor[(fx, y, fz)] for fx, fz in footprint if (fx, y, fz) in qcolor}
                if len(shown_colors) > 1:
                    continue
                # Bigger bricks first; among near-equal sizes prefer the one
                # that ties together more distinct bricks in the layer below.
                below = {owner.get((fx, y - 1, fz)) for fx, fz in footprint} - {None}
                score = w * d + BOND_WEIGHT * len(below)
                if score > best_score:
                    color = shown_colors.pop() if shown_colors else interior_color
                    best, best_score = (w, d, footprint, color), score
            w, d, footprint, (color_name, color_hex) = best   # 1x1 always fits
            bricks.append({
                "x": x, "y": y, "z": z,
                "width": w, "depth": d, "height": 1,
                "type": f"{min(w, d)}x{max(w, d)}",
                "color_name": color_name,
                "color": color_hex,
            })
            for fx, fz in footprint:
                owner[(fx, y, fz)] = len(bricks) - 1
            remaining.difference_update(footprint)

    return bricks


def _build_parts_list(bricks: list[dict]) -> list[dict]:
    counts: dict[tuple, int] = {}
    for b in bricks:
        key = (b["type"], b["color_name"], b["color"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"type": t, "color_name": cn, "color": c, "count": n}
        for (t, cn, c), n in sorted(counts.items())
    ]


async def run_lego_conversion(run_id: str) -> dict:
    db = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("voxelized",):
        raise HTTPException(status_code=409, detail=f"Run is in status '{run['status']}', expected 'voxelized'")

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "converting", "lego_started_at": datetime.now(timezone.utc)}},
    )

    try:
        # Load voxels from GridFS
        voxel_stream = await gridfs.open_download_stream(
            ObjectId(run["voxelization"]["voxel_file_id"])
        )
        voxels = json.loads(await voxel_stream.read())

        bricks = _pack_bricks(voxels)
        parts_list = _build_parts_list(bricks)

        # Store final model JSON (used by Three.js renderer)
        model_data = {
            "bricks": bricks,
            "dimensions": {
                "width": max((b["x"] + b["width"]) for b in bricks) if bricks else 0,
                "height": max((b["y"] + b["height"]) for b in bricks) if bricks else 0,
                "depth": max((b["z"] + b["depth"]) for b in bricks) if bricks else 0,
            },
        }
        model_bytes = json.dumps(model_data).encode()
        model_file_id = await gridfs.upload_from_stream(
            "model.json",
            io.BytesIO(model_bytes),
            metadata={"run_id": run_id, "content_type": "application/json"},
        )

        result = {
            "model_file_id": str(model_file_id),
            "brick_count": len(bricks),
            "parts_list": parts_list,
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {
                "$set": {
                    "status": "complete",
                    "lego": result,
                    "lego_completed_at": datetime.now(timezone.utc),
                }
            },
        )
        return {"run_id": run_id, "status": "complete", **result}

    except Exception as exc:
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
