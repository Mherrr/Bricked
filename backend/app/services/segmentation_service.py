"""
YOLO segmentation stage.

For each uploaded image:
  1. Pre-process (CLAHE contrast + unsharp sharpening) to maximise YOLO confidence
  2. Run YOLOv11-seg to detect all objects and their masks
  3. Pick the most prominent object (largest mask area), then merge in any
     detections touching it — a straw in a cup is reported separately
  4. Discard images whose mask fill ratio falls outside acceptable bounds
     Fill ratio = mask pixels inside bounding box / bounding box area.
     Too low → fragmented / missed object.  Too high → bbox flooded w/ background.
  5. Apply the mask — background becomes transparent (RGBA PNG)
  6. Store the masked image back to GridFS
  7. Update the run document with segmented image references
"""
import io
import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from functools import lru_cache
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

MODEL_NAME     = "yolo11x-seg.pt"
CONF_THRESHOLD = 0.35   # initial detection threshold
CONF_RETRY     = 0.10   # fallback threshold if nothing found at CONF_THRESHOLD

# Bounding-box fill ratio gate:
#   fill_ratio = (mask pixels inside bbox) / (bbox area)
# A well-segmented object should fill a meaningful but not total fraction of
# its own bounding box.  Values outside this range indicate either a shredded
# mask (too low) or a flood-fill that captured the background (too high).
MIN_FILL_RATIO = 0.30   # below → mask too sparse / object not covered
MAX_FILL_RATIO = 0.98   # above → mask swallowed the whole bbox region

# YOLO reports a cup and the straw standing in it as two separate objects.
# Detections whose boxes touch (or sit within this fraction of the image's
# longer side of) the primary detection are merged into one silhouette, so
# thin attached parts survive instead of being dropped with the smaller mask.
MERGE_MARGIN_FRAC = 0.02


@lru_cache(maxsize=1)
def _load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_NAME)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    import cv2
    from PIL import Image

    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if bgr is None:
        raise ValueError("Could not decode image — file may be corrupt")

    return bgr


def _preprocess_for_detection(bgr: np.ndarray) -> np.ndarray:
    """
    Enhance contrast and sharpness so YOLO receives a cleaner signal.

    Steps:
      1. CLAHE on the L channel (LAB) — lifts local contrast without
         blowing out highlights, improving detection on low-contrast backgrounds.
      2. Unsharp mask — amplifies edges so the model sees crisper boundaries,
         improving mask precision and bounding box tightness.
    """
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blurred   = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)

    return sharpened


def _compute_fill_ratio(
    binary_mask: np.ndarray,
    box_xyxy: list[float],
) -> float:
    """
    Compute what fraction of the bounding box is covered by the mask.

    binary_mask : H×W uint8, values 0 or 255, in full image coordinates
    box_xyxy    : [x1, y1, x2, y2] in image pixel coordinates
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(binary_mask.shape[1], x2)
    y2 = min(binary_mask.shape[0], y2)

    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    mask_in_bbox = binary_mask[y1:y2, x1:x2]
    fill = float((mask_in_bbox > 0).sum()) / bbox_area
    return fill


def _boxes_touch(a: list[float], b: list[float], margin: float) -> bool:
    """True when two xyxy boxes overlap, or lie within `margin` px of each other."""
    return not (
        a[2] + margin < b[0] or b[2] + margin < a[0]
        or a[3] + margin < b[1] or b[3] + margin < a[1]
    )


def _merge_touching_masks(
    masks: np.ndarray,
    boxes: list[list[float]],
    best_idx: int,
    shape: tuple[int, int],
    margin: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    """
    Union the primary detection with every mask whose box touches it, transitively.

    Taking only the largest mask discards parts the detector reports separately —
    a straw in a cup, a handle, a lid — amputating real geometry before the
    silhouette ever reaches reconstruction.  Growing the selection by adjacency
    keeps the whole physical object while still ignoring unrelated items
    elsewhere in the frame.

    Returns (binary_mask 0/255, merged xyxy box, indices merged).
    """
    import cv2

    h, w = shape
    keep: set[int] = {best_idx}
    merged = list(boxes[best_idx])

    changed = True
    while changed:                      # transitive: straw → lid → cup
        changed = False
        for i, b in enumerate(boxes):
            if i in keep or not _boxes_touch(merged, b, margin):
                continue
            keep.add(i)
            merged = [min(merged[0], b[0]), min(merged[1], b[1]),
                      max(merged[2], b[2]), max(merged[3], b[3])]
            changed = True

    union = np.zeros((h, w), dtype=np.uint8)
    for i in sorted(keep):
        resized = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        union |= (resized > 0.5).astype(np.uint8)

    return union * 255, merged, sorted(keep)


def _segment_image(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Pre-process then run YOLO segmentation on raw image bytes.

    Returns:
        masked_png_bytes : RGBA PNG with background removed
        meta             : fill_ratio, confidence, bounding box
    Raises:
        ValueError if no object is detected or mask quality is unacceptable.
    """
    import cv2
    from PIL import Image

    model = _load_model()
    bgr   = _decode_image(image_bytes)
    h, w  = bgr.shape[:2]

    bgr_proc = _preprocess_for_detection(bgr)

    result = model(bgr_proc, conf=CONF_THRESHOLD, verbose=False)[0]

    if result.masks is None or len(result.masks) == 0:
        result = model(bgr_proc, conf=CONF_RETRY, verbose=False)[0]

    if result.masks is None or len(result.masks) == 0:
        raise ValueError(
            "No objects detected — ensure the object is clearly visible, "
            "well-lit, and contrasts with the background"
        )

    masks    = result.masks.data.cpu().numpy()   # (N, H_mask, W_mask) float32
    areas    = [m.sum() for m in masks]
    best_idx = int(np.argmax(areas))

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        conf  = float(result.boxes.conf[best_idx].cpu())
    else:
        boxes = [[0.0, 0.0, float(w), float(h)] for _ in masks]
        conf  = 0.0

    # Merge parts the detector split off (straw, handle, lid) back into the object
    margin = MERGE_MARGIN_FRAC * float(max(h, w))
    binary_mask, box, merged = _merge_touching_masks(masks, boxes, best_idx, (h, w), margin)

    fill_ratio = _compute_fill_ratio(binary_mask, box)

    if fill_ratio < MIN_FILL_RATIO:
        raise ValueError(
            f"Mask fill ratio too low ({fill_ratio:.2f} < {MIN_FILL_RATIO}) — "
            "the segmentation mask covers too little of the detected region; "
            "use a contrasting background or improve lighting"
        )
    if fill_ratio > MAX_FILL_RATIO:
        raise ValueError(
            f"Mask fill ratio too high ({fill_ratio:.2f} > {MAX_FILL_RATIO}) — "
            "the mask likely flooded into the background; "
            "use a plain background that contrasts with the object"
        )

    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba    = np.dstack([rgb, binary_mask])
    pil_img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    return buf.getvalue(), {
        "fill_ratio": round(fill_ratio, 4),
        "confidence": round(conf, 4),
        "box": box,
        "merged_masks": len(merged),
    }


async def run_segmentation(run_id: str) -> dict:
    db      = get_db()
    gridfs  = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("uploaded",):
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run['status']}', expected 'uploaded'",
        )

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "segmenting", "segmentation_started_at": datetime.now(timezone.utc)}},
    )

    try:
        segmented: list[dict] = []
        skipped:   list[dict] = []

        for view_index, img in enumerate(run.get("images", [])):
            stream      = await gridfs.open_download_stream(ObjectId(img["file_id"]))
            image_bytes = await stream.read()

            try:
                masked_bytes, detection_meta = await asyncio.get_event_loop().run_in_executor(
                    None, _segment_image, image_bytes
                )
            except ValueError as skip_exc:
                reason = str(skip_exc)
                logger.warning("Skipping %s: %s", img["filename"], reason)
                skipped.append({"filename": img["filename"], "reason": reason})
                continue

            seg_filename = img["filename"].rsplit(".", 1)[0] + "_seg.png"
            seg_file_id  = await gridfs.upload_from_stream(
                seg_filename,
                io.BytesIO(masked_bytes),
                metadata={"run_id": run_id, "content_type": "image/png", "stage": "segmented"},
            )

            logger.info(
                "Segmented %s  fill=%.2f  conf=%.2f",
                img["filename"],
                detection_meta["fill_ratio"],
                detection_meta["confidence"],
            )
            segmented.append({
                "view_index":        view_index,   # position in the capture sequence
                "original_file_id":  img["file_id"],
                "segmented_file_id": str(seg_file_id),
                "filename":          seg_filename,
                "detection":         detection_meta,
            })

        if len(segmented) < 2:
            raise ValueError(
                f"Only {len(segmented)} image(s) passed the quality gate "
                f"(need at least 2). Skipped images: "
                + ", ".join(f"{s['filename']} ({s['reason']})" for s in skipped)
            )

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status":                    "segmented",
                "segmented_images":          segmented,
                "skipped_images":            skipped,
                "segmentation_completed_at": datetime.now(timezone.utc),
            }},
        )
        return {
            "run_id":           run_id,
            "status":           "segmented",
            "segmented_images": segmented,
            "skipped_images":   skipped,
        }

    except HTTPException:
        raise
    except Exception as exc:
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
