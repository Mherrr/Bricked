"""
Unit tests for the pipeline's pure functions (no MongoDB, no YOLO weights).

    cd backend && python -m pytest -q
"""
import cv2
import numpy as np
import pytest

from app.services import lego_service, reconstruction_service as R, segmentation_service as S, voxel_service


# ── Helpers ───────────────────────────────────────────────────────────────────

def _project(point, angle, elevation, size=(480, 640), f=500.0):
    """Pinhole-project a world point with the backend's turntable camera."""
    Rm, t = R._build_camera(angle, elevation, R.CAMERA_DISTANCE)
    pc = Rm @ np.asarray(point, float) + t
    h, w = size
    return f * pc[0] / pc[2] + w / 2, f * pc[1] / pc[2] + h / 2


def _box_silhouettes(half=(0.6, 0.9, 0.3), n=8, elevation=R.CAMERA_ELEVATION, size=(480, 640)):
    """Exact silhouettes of an axis-aligned box seen from n turntable cameras."""
    hx, hy, hz = half
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    sils = []
    for i in range(n):
        a = 2 * np.pi * i / n
        uv = np.array([_project(c, a, elevation, size) for c in corners], np.float32)
        m = np.zeros(size, np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.int32)), 1)
        sils.append(m)
    return sils


def _iou_box(pts, half):
    """IoU of the (filled) carved hull against the box, both bbox-normalised."""
    from scipy.ndimage import binary_fill_holes
    res = 48
    lo, hi = pts.min(0), pts.max(0)
    n = (pts - (lo + hi) / 2) / (np.max(hi - lo) / 2)
    idx = np.clip(((n + 1) / 2 * res).astype(int), 0, res - 1)
    occ = np.zeros((res,) * 3, bool)
    occ[tuple(idx.T)] = True
    occ = binary_fill_holes(occ)
    g = np.array(half) / max(half)
    c = (np.arange(res) + 0.5) / res * 2 - 1
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    gt = (abs(X) <= g[0]) & (abs(Y) <= g[1]) & (abs(Z) <= g[2])
    return (occ & gt).sum() / (occ | gt).sum()


# ── Reconstruction ────────────────────────────────────────────────────────────

def test_camera_matches_pixel_axes():
    # From the front camera, world +X is image-right and world +Y is image-up
    u0, v0 = _project([0, 0, 0], 0.0, 0.0)
    ur, _ = _project([0.5, 0, 0], 0.0, 0.0)
    _, vu = _project([0, 0.5, 0], 0.0, 0.0)
    assert ur > u0
    assert vu < v0


def test_view_poses_keep_gaps_for_skipped_photos():
    angles, elev = R.view_poses([0, 1, 3, 4, 5, 6, 7], 8)   # photo 2 was rejected
    assert np.allclose(angles, 2 * np.pi * np.array([0, 1, 3, 4, 5, 6, 7]) / 8)
    assert np.allclose(elev, R.CAMERA_ELEVATION)


def test_view_poses_two_ring_capture():
    angles, elev = R.view_poses(list(range(16)), 16)
    assert np.allclose(angles[:8], angles[8:])
    assert np.allclose(elev[:8], R.CAMERA_ELEVATION)
    assert np.allclose(elev[8:], R.HIGH_ELEVATION)


def test_crop_windows_share_one_scale():
    a = np.zeros((200, 300), np.uint8); a[50:150, 100:140] = 1   # narrow view
    b = np.zeros((200, 300), np.uint8); b[50:150, 60:220] = 1    # wide view
    for mode in ("shared_crop", "shared_scale"):
        wa, wb = R._crop_windows([a, b], mode=mode)
        assert (wa[1] - wa[0]) == (wb[1] - wb[0])


def test_visual_hull_recovers_elongated_box():
    half = (0.6, 0.9, 0.3)
    sils = _box_silhouettes(half)
    pts, colors = R.carve_views(list(range(8)), 8, sils, [None] * 8)
    assert len(pts) == len(colors) > 0
    assert _iou_box(pts, half) > 0.8


def test_to_point_list_round_trip():
    pts = np.array([[0.1, -0.2, 0.3]], np.float32)
    cols = np.array([[1, 2, 3]], np.uint8)
    assert R.to_point_list(pts, cols) == [{"x": 0.1, "y": -0.2, "z": 0.3, "r": 1, "g": 2, "b": 3}]


# ── Segmentation ──────────────────────────────────────────────────────────────

def test_backdrop_segmentation_on_gradient_backdrop():
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = (200 - 40 * yy / h)[..., None] * np.array([1.0, 1.0, 1.02])
    img = bg.copy()
    gt = ((xx - 320) ** 2 + (yy - 240) ** 2) < 110 ** 2
    img[gt] = (40, 90, 200)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    mask, _ = S._backdrop_mask(img)
    m = mask > 0
    assert (m & gt).sum() / (m | gt).sum() > 0.97


def test_backdrop_keeps_through_holes():
    h, w = 400, 400
    img = np.full((h, w, 3), 210, np.uint8)
    cv2.circle(img, (200, 200), 120, (30, 30, 160), thickness=40)   # a ring
    mask, _ = S._backdrop_mask(img)
    assert mask[200, 200] == 0          # backdrop visible through the ring stays background


def test_busy_background_is_not_a_backdrop():
    rng = np.random.default_rng(1)
    img = cv2.resize(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8), (640, 480),
                     interpolation=cv2.INTER_NEAREST)
    assert S._backdrop_mask(img) is None


# ── Voxelization ──────────────────────────────────────────────────────────────

def test_voxelization_fills_a_hollow_shell():
    # Surface of a sphere, as the reconstruction stage now stores it
    g = np.linspace(-1, 1, 64)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    shell = np.abs(r - 0.8) < 0.035
    pts = [{"x": x, "y": y, "z": z, "r": 200, "g": 30, "b": 30}
           for x, y, z in zip(X[shell], Y[shell], Z[shell])]
    vox = voxel_service._build_voxel_grid(pts)
    n = voxel_service.TARGET_STUDS                      # sphere diameter in studs
    solid = 4 / 3 * np.pi * (n / 2) ** 2 * (n / 2 / voxel_service.BRICK_ASPECT)
    assert len(vox) > 0.8 * solid                       # interior was filled
    assert all((v["r"], v["g"], v["b"]) != (128, 128, 128) for v in vox)
    xs, ys = [v["x"] for v in vox], [v["y"] for v in vox]
    assert max(xs) - min(xs) + 1 == pytest.approx(n, abs=2)   # sized to TARGET_STUDS
    assert max(ys) - min(ys) + 1 == pytest.approx(n / voxel_service.BRICK_ASPECT, abs=2)


# ── LEGO conversion ───────────────────────────────────────────────────────────

def test_palette_hex_matches_rgb():
    for name, (hx, r, g, b) in lego_service.LEGO_PALETTE.items():
        assert hx == f"#{r:02X}{g:02X}{b:02X}", name


def test_pack_bricks_covers_every_voxel_once():
    rng = np.random.default_rng(2)
    voxels = []
    for x in range(10):
        for y in range(3):
            for z in range(7):
                if rng.random() < 0.8:
                    c = (201, 26, 9) if x < 5 else (0, 85, 191)
                    voxels.append({"x": x, "y": y, "z": z, "r": c[0], "g": c[1], "b": c[2]})
    bricks = lego_service._pack_bricks(voxels)
    covered = []
    for b in bricks:
        assert tuple(sorted((b["width"], b["depth"]))) in lego_service.BRICK_TYPES
        covered += [(b["x"] + dx, b["y"], b["z"] + dz) for dx in range(b["width"]) for dz in range(b["depth"])]
    assert sorted(covered) == sorted((v["x"], v["y"], v["z"]) for v in voxels)
    assert {b["color_name"] for b in bricks} == {"Bright Red", "Bright Blue"}
    assert len(bricks) < len(voxels) / 2                  # merging actually happens
