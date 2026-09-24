"""
End-to-end API benchmark: upload each scene's photos to a running backend,
trigger every stage in order, and record wall-clock latency per request plus
the size of the stored point cloud and the final model.

    uvicorn app.main:app --port 8000        # in backend/, with MongoDB running
    python e2e_api.py /path/to/scenes --api http://localhost:8000 --json e2e.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
from pymongo import MongoClient
from bson import ObjectId


def run_one(client: httpx.Client, scene: Path, db) -> dict:
    files = [("images", (p.name, p.read_bytes(), "image/jpeg")) for p in sorted((scene / "images").glob("*.jpg"))]
    r = {"name": scene.name, "views": len(files)}

    t0 = time.perf_counter()
    res = client.post("/api/uploads/runs", files=files)
    res.raise_for_status()
    run_id = res.json()["run_id"]
    r["t_upload"] = time.perf_counter() - t0

    for stage in ("segment", "reconstruct", "voxelize", "lego"):
        t = time.perf_counter()
        res = client.post(f"/api/runs/{run_id}/{stage}")
        r[f"t_{stage}"] = time.perf_counter() - t
        if res.status_code != 202:
            r["failed"] = f"{stage}: {res.status_code} {res.text[:200]}"
            return r
    r["t_total"] = time.perf_counter() - t0

    for path in ("pointcloud", "voxels", "model", "parts"):
        t = time.perf_counter()
        res = client.get(f"/api/runs/{run_id}/{path}")
        res.raise_for_status()
        r[f"t_get_{path}"] = time.perf_counter() - t
        r[f"bytes_{path}"] = len(res.content)

    run = db.runs.find_one({"_id": ObjectId(run_id)})
    cloud = db["images.files"].find_one({"_id": ObjectId(run["reconstruction"]["point_cloud_file_id"])})
    r["point_cloud_mb"] = cloud["length"] / 1e6
    r["points"] = run["reconstruction"]["point_count"]
    r["bricks"] = run["lego"]["brick_count"]
    r["segmented"] = len(run["segmented_images"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", type=Path)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--mongo", default="mongodb://localhost:27017")
    ap.add_argument("--db", default="bricked")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    db = MongoClient(args.mongo)[args.db]
    rows = []
    with httpx.Client(base_url=args.api, timeout=900) as client:
        for scene in sorted(p for p in args.scenes.iterdir() if (p / "images").is_dir()):
            if args.only and scene.name not in args.only:
                continue
            r = run_one(client, scene, db)
            rows.append(r)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    ok = [r for r in rows if "failed" not in r]
    keys = [k for k in ok[0] if k.startswith(("t_", "bytes_", "point_cloud_mb", "points", "bricks"))] if ok else []
    summary = {"runs": len(rows), "completed": len(ok),
               **{k: statistics.mean(r[k] for r in ok) for k in keys}}
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))


if __name__ == "__main__":
    main()
