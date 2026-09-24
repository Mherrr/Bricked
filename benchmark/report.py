"""
Print a before/after markdown table from evaluate.py / e2e_api.py result files.

    python report.py --before orig8.json --after new8.json
"""
import argparse
import json
from pathlib import Path

ROWS = [
    ("seg_pass_rate", "Photos segmented (pass rate)", "{:.0%}"),
    ("seg_mask_iou", "Segmentation mask IoU", "{:.3f}"),
    ("iou3d_hull", "3D IoU — carved hull (mean)", "{:.3f}"),
    ("iou3d_hull_median", "3D IoU — carved hull (median)", "{:.3f}"),
    ("iou3d_voxels", "3D IoU — final voxel model", "{:.3f}"),
    ("voxels", "Voxels per model", "{:,.0f}"),
    ("bricks", "Bricks per model", "{:,.0f}"),
    ("studs_per_brick", "Studs per brick", "{:.2f}"),
    ("bond_ratio", "Bricks bonded to ≥2 below", "{:.0%}"),
    ("connected", "Bricks in largest connected build", "{:.1%}"),
    ("colors", "LEGO colours used", "{:.1f}"),
    ("delta_e", "Colour error ΔE (visible voxels)", "{:.1f}"),
    ("t_segment", "Segmentation time (s)", "{:.2f}"),
    ("t_reconstruct", "Reconstruction time (s)", "{:.2f}"),
    ("t_voxelize", "Voxelization time (s)", "{:.2f}"),
    ("t_lego", "Brick packing time (s)", "{:.2f}"),
    ("t_total", "Pipeline compute time (s)", "{:.2f}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    args = ap.parse_args()
    b = json.loads(args.before.read_text())["summary"]
    a = json.loads(args.after.read_text())["summary"]
    print("| Metric | Before | After |")
    print("| --- | --- | --- |")
    for key, label, fmt in ROWS:
        if key in a:
            before = fmt.format(b[key]) if key in b else "—"
            print(f"| {label} | {before} | {fmt.format(a[key])} |")


if __name__ == "__main__":
    main()
