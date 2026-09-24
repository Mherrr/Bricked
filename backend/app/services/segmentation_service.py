"""
Segmentation stage.

For each uploaded image:
  1. Backdrop segmentation (preferred).  Bricked asks for a plain backdrop, so
     the backdrop is modelled from the image border — a smooth quadratic
     colour surface in CIE-LAB, which absorbs lighting gradients and vignetting
     — and every pixel that departs from it is foreground.  The coarse mask is
     refined with GrabCut.  This is class-agnostic: it works for objects YOLO
     has no COCO class for (an avocado, a boom box, a lantern).
  2. YOLO fallback.  When the border is too busy to model as a backdrop, run
     YOLO11-seg (after CLAHE + unsharp pre-processing), take the largest
     instance, and gate it on bounding-box fill ratio:
       fill ratio = mask pixels inside bounding box / bounding box area.
       Too low → fragmented / missed object.  Too high → bbox flooded w/ background.
  3. Apply the mask — background becomes transparent (RGBA PNG)
  4. Store the masked image back to GridFS
  5. Update the run document with segmented image references
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
MIN_FILL_RATIO = 0.40   # below → mask too sparse / object not covered
MAX_FILL_RATIO = 0.98   # above → mask swallowed the whole bbox region


# Backdrop model
BORDER_FRAC      = 0.04   # width of the image border sampled as backdrop
BACKDROP_MAX_MAD = 4.0    # border residual (LAB units) above which the backdrop is "busy"
FG_SIGMAS        = 6.0    # foreground = residual above this many robust sigmas…
FG_MIN_DELTA     = 8.0    # …and at least this many LAB units from the backdrop
MIN_AREA_FRAC    = 0.005  # reject masks smaller than this fraction of the image
MAX_AREA_FRAC    = 0.90   # …or larger than this
GRABCUT_SIDE     = 640    # GrabCut runs at this resolution (long side) for speed


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


def _backdrop_mask(bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    Class-agnostic foreground mask for an object on a plain backdrop.

    Returns (mask uint8 0/255, backdrop residual MAD) or None when the image
    border is too busy to be a plain backdrop.
    """
    import cv2
    from scipy.ndimage import binary_fill_holes

    h, w = bgr.shape[:2]
    scale = min(1.0, GRABCUT_SIDE / max(h, w))
    small = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Fit a quadratic colour surface to the border pixels, per LAB channel
    b = max(2, int(round(min(sh, sw) * BORDER_FRAC)))
    border = np.zeros((sh, sw), dtype=bool)
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    yn, xn = yy / sh - 0.5, xx / sw - 0.5
    basis = np.stack([np.ones_like(xn), xn, yn, xn * xn, yn * yn, xn * yn], axis=-1)
    A = basis[border]
    coef, *_ = np.linalg.lstsq(A, lab[border], rcond=None)
    backdrop = basis @ coef                                   # (sh, sw, 3)

    resid = np.linalg.norm(lab - backdrop, axis=-1)
    br = resid[border]
    mad = float(np.median(np.abs(br - np.median(br)))) * 1.4826
    # Busy border (clutter, or the object itself running off-frame) → not a backdrop
    if mad > BACKDROP_MAX_MAD or (br > FG_MIN_DELTA * 2).mean() > 0.10:
        return None

    thresh = max(FG_SIGMAS * mad + float(np.median(br)), FG_MIN_DELTA)
    coarse = (resid > thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, k)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, k)
    if coarse.sum() < MIN_AREA_FRAC * sh * sw:
        return None

    # GrabCut: confident pixels seed the colour models, the band between refines
    gc = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
    gc[resid < 0.5 * thresh] = cv2.GC_BGD
    gc[coarse > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(coarse, k, iterations=2) > 0] = cv2.GC_FGD
    gc[border] = cv2.GC_BGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        refined = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        refined = coarse

    # Keep the main object: largest component plus any sizeable pieces
    n, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= 0.05 * areas.max())[0]
    obj = np.isin(labels, keep)
    # Fill pin-holes, but leave genuine openings (a mug handle, a watch strap)
    # where the backdrop shows through
    holes = binary_fill_holes(obj) & ~obj
    n_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    small = 1 + np.where(h_stats[1:, cv2.CC_STAT_AREA] < 0.01 * obj.sum())[0]
    refined = (obj | np.isin(h_labels, small)).astype(np.uint8)

    mask = cv2.resize(refined * 255, (w, h), interpolation=cv2.INTER_LINEAR)
    return (mask > 127).astype(np.uint8) * 255, mad


def _encode_rgba(bgr: np.ndarray, binary_mask: np.ndarray) -> bytes:
    import cv2
    from PIL import Image

    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, binary_mask])
    buf  = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _segment_image(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Segment the object in raw image bytes — backdrop model first, YOLO fallback.

    Returns:
        masked_png_bytes : RGBA PNG with background removed
        meta             : method, fill_ratio, confidence, bounding box
    Raises:
        ValueError if no object is detected or mask quality is unacceptable.
    """
    import cv2

    bgr   = _decode_image(image_bytes)
    h, w  = bgr.shape[:2]

    backdrop = _backdrop_mask(bgr)
    if backdrop is not None:
        binary_mask, mad = backdrop
        area = (binary_mask > 0).mean()
        if MIN_AREA_FRAC <= area <= MAX_AREA_FRAC:
            ys, xs = np.where(binary_mask > 0)
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            return _encode_rgba(bgr, binary_mask), {
                "method":     "backdrop",
                "fill_ratio": round(_compute_fill_ratio(binary_mask, box), 4),
                "confidence": round(float(np.clip(1.0 - mad / BACKDROP_MAX_MAD, 0.0, 1.0)), 4),
                "box":        box,
            }

    model = _load_model()

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

    best_mask    = masks[best_idx]
    mask_resized = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary_mask  = (mask_resized > 0.5).astype(np.uint8) * 255

    box = result.boxes.xyxy[best_idx].cpu().tolist() if result.boxes is not None else [0, 0, w, h]
    conf = float(result.boxes.conf[best_idx].cpu()) if result.boxes is not None else 0.0

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

    return _encode_rgba(bgr, binary_mask), {
        "method":     "yolo",
        "fill_ratio": round(fill_ratio, 4),
        "confidence": round(conf, 4),
        "box": box,
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
