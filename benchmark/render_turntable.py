"""
Render synthetic turntable captures of textured glTF models, with ground truth.

For each model this writes:
    <out>/<name>/images/NN.jpg   photo-like RGB captures (plain studio background)
    <out>/<name>/masks/NN.png    ground-truth foreground masks
    <out>/<name>/gt.npz          ground-truth solid occupancy (64^3 in [-1,1]^3)
                                 plus per-voxel surface colours

Cameras follow the same turntable model the backend assumes: evenly spaced in
azimuth on a circle of radius CAMERA_DISTANCE around the Y axis, looking at the
origin.  `--rings` adds an optional second ring from higher up (the "8 level +
8 from above" capture recommended in the UI).

Needs pyrender + an OSMesa build of Mesa:
    PYOPENGL_PLATFORM=osmesa python render_turntable.py models/*.glb --out scenes
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import pyrender
import trimesh
from PIL import Image

CAMERA_DISTANCE = 3.0
GT_RES = 64


def look_at(cam_pos: np.ndarray) -> np.ndarray:
    """OpenGL camera pose (camera looks down its local -Z, +Y up)."""
    z = cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, cam_pos
    return pose


def load_normalised(path: Path) -> trimesh.Scene:
    scene = trimesh.load(str(path), force="scene")
    lo, hi = scene.bounds
    centre = (lo + hi) / 2.0
    scale = 1.0 / (np.max(hi - lo) / 2.0)          # fit in [-1, 1]^3
    T = np.eye(4)
    T[:3, :3] *= scale
    T[:3, 3] = -centre * scale
    scene.apply_transform(T)
    # pyrender has no image-based lighting, so metallic PBR surfaces render
    # black.  Treat everything as a dielectric so textures read like a photo.
    for geom in scene.geometry.values():
        mat = getattr(getattr(geom, "visual", None), "material", None)
        if mat is not None and hasattr(mat, "metallicFactor"):
            mat.metallicFactor = 0.0
            mat.roughnessFactor = max(0.6, mat.roughnessFactor or 0.0)
    return scene


def ground_truth(scene: trimesh.Scene) -> dict:
    """Solid occupancy on a GT_RES^3 grid over [-1,1]^3.

    Most scanned/artist meshes are not watertight, so instead of trusting
    `contains`, rasterise the surface, close pin-holes, flood-fill from the
    outside, and treat everything unreachable as solid.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure

    mesh = scene.to_geometry()
    pitch = 2.0 / GT_RES
    pts, _ = trimesh.sample.sample_surface_even(mesh, 400_000) if len(mesh.faces) else (np.empty((0, 3)), None)
    pts = np.vstack([pts, mesh.vertices])
    idx = np.clip(np.floor((pts + 1.0) / pitch).astype(int), 0, GT_RES - 1)
    shell = np.zeros((GT_RES,) * 3, dtype=bool)
    shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    pad = np.pad(shell, 2)
    closed = binary_closing(pad, structure=generate_binary_structure(3, 1), iterations=1)
    # Objects stand on a turntable: seal the floor so open-bottomed shapes
    # (draped cloth, upturned bowls) count as solid down to where they rest.
    floor = int(np.argwhere(closed.any(axis=(0, 2)))[0, 0])
    closed[:, floor, :] |= binary_fill_holes(closed[:, floor, :])
    occ = binary_fill_holes(closed)[2:-2, 2:-2, 2:-2] | shell
    return {"occupancy": occ}


def render_scene(scene: trimesh.Scene, out: Path, n_views: int, rings: list[float],
                 width: int, height: int, yfov: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    bg = np.array([0.80, 0.80, 0.82, 1.0])
    pscene = pyrender.Scene.from_trimesh_scene(scene, bg_color=bg, ambient_light=[0.35] * 3)
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / height)
    cam_node = pscene.add(cam, pose=np.eye(4))
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    key_node = pscene.add(key, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(width, height)

    view = 0
    for elev in rings:
        for i in range(n_views):
            a = 2.0 * np.pi * i / n_views
            cam_pos = CAMERA_DISTANCE * np.array(
                [np.cos(elev) * np.sin(a), np.sin(elev), np.cos(elev) * np.cos(a)]
            )
            pose = look_at(cam_pos)
            pscene.set_pose(cam_node, pose)
            # Key light slightly above-left of the camera, like a softbox
            light_pose = look_at(cam_pos + np.array([0.0, 1.5, 0.0]) + pose[:3, 0] * -1.0)
            pscene.set_pose(key_node, light_pose)

            color, depth = renderer.render(pscene)
            mask = depth > 0
            # Studio backdrop that is not perfectly flat: a lighting gradient,
            # vignette and a random tint per shot, plus sensor noise.
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            r2 = ((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2
            shade = (1.05 - 0.25 * yy / height) * (1.0 - 0.6 * r2)
            tint = bg[:3] * 255 * rng.uniform(0.92, 1.05, 3)
            backdrop = shade[..., None] * tint[None, None, :]
            color = np.where(mask[..., None], color.astype(np.float32), backdrop)
            noisy = np.clip(color + rng.normal(0, 2.5, color.shape), 0, 255)
            Image.fromarray(noisy.astype(np.uint8)).save(out / "images" / f"{view:02d}.jpg", quality=92)
            Image.fromarray((mask * 255).astype(np.uint8)).save(out / "masks" / f"{view:02d}.png")
            view += 1
    renderer.delete()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("scenes"))
    ap.add_argument("--views", type=int, default=8, help="views per ring")
    ap.add_argument("--rings", type=float, nargs="+", default=[0.3],
                    help="camera elevation (radians) of each ring")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--gt-only", action="store_true", help="only recompute gt.npz")
    ap.add_argument("--yfov", type=float, default=np.deg2rad(50))
    args = ap.parse_args()

    for k, path in enumerate(args.models):
        name = path.stem
        out = args.out / name
        scene = load_normalised(path)
        if args.gt_only:
            gt = ground_truth(scene)
            np.savez_compressed(out / "gt.npz", **gt)
            print(f"{name}: gt voxels={int(gt['occupancy'].sum())}")
            continue
        try:
            render_scene(scene, out, args.views, args.rings, args.width, args.height, args.yfov, seed=k)
        except Exception as exc:                      # unsupported material etc.
            print(f"{name}: skipped ({type(exc).__name__}: {exc})")
            continue
        gt = ground_truth(scene)
        np.savez_compressed(out / "gt.npz", **gt)
        print(f"{name}: {len(args.rings) * args.views} views, gt voxels={int(gt['occupancy'].sum())}")


if __name__ == "__main__":
    main()
