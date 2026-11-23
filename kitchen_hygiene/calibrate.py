"""Empty-scene calibration for the pest detector.

At installation, the camera is pointed at the kitchen with nobody in it, just like marking the zones.
Anything the pest detector "finds" in that empty scene is a look-alike built into this camera view
(a stove knob, a drain, a dark corner). Those spots are remembered, and pest detections inside them
are ignored afterwards. A real pest is still caught as it moves through the rest of the kitchen.

We calibrate on the starting frame of every clip: all our clips start from the same empty kitchen
picture, so a spot that is flagged in the starting frame of at least `min_votes` clips is a fixed
look-alike, not a real pest that happened to be there once.
"""
import cv2
import numpy as np


def starting_frames(video_paths):
    frames = []
    for p in video_paths:
        cap = cv2.VideoCapture(str(p))
        ok, f = cap.read()
        cap.release()
        if ok:
            frames.append(f)
    return frames


def _detect(models, img, pest_classes, conf, imgsz):
    h, w = img.shape[:2]
    out = []
    for m in models:
        r = m.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        if r.boxes is None:
            continue
        for b, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if m.names[int(k)] in pest_classes:
                x1, y1, x2, y2 = map(float, b)
                out.append((x1 / w, y1 / h, x2 / w, y2 / h))
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cluster_spots(per_image_boxes, min_votes=2, margin=0.3, link_iou=0.2):
    """Group boxes found in different images at the same place; keep groups seen in >= min_votes images.
    Returns normalised (x1, y1, x2, y2) boxes, each enlarged by `margin` of its size on every side."""
    clusters = []                                   # each: {"box": union box, "images": set}
    for i, boxes in enumerate(per_image_boxes):
        for b in boxes:
            for c in clusters:
                if _iou(b, c["box"]) >= link_iou:
                    cb = c["box"]
                    c["box"] = (min(cb[0], b[0]), min(cb[1], b[1]), max(cb[2], b[2]), max(cb[3], b[3]))
                    c["images"].add(i)
                    break
            else:
                clusters.append({"box": b, "images": {i}})
    spots = []
    for c in clusters:
        if len(c["images"]) >= min(min_votes, len(per_image_boxes)):
            x1, y1, x2, y2 = c["box"]
            mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
            spots.append((max(0.0, x1 - mw), max(0.0, y1 - mh), min(1.0, x2 + mw), min(1.0, y2 + mh)))
    return spots


def calibrate_pests(video_paths, cfg, conf=0.15, min_votes=2, margin=0.3):
    """Returns (spots, preview_image). Put the spots in cfg.pest_ignore_boxes."""
    from ultralytics import YOLO
    models = [YOLO(p) for p in cfg.custom_models]
    frames = starting_frames(video_paths)
    if not models or not frames:
        return [], None
    per_image = [_detect(models, f, cfg.pest_classes, conf, cfg.imgsz) for f in frames]
    spots = cluster_spots(per_image, min_votes=min_votes, margin=margin)
    preview = frames[0].copy()
    h, w = preview.shape[:2]
    for (x1, y1, x2, y2) in spots:
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(preview, p1, p2, (0, 0, 255), 3)
        cv2.putText(preview, "ignored look-alike", (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return spots, preview


def in_ignored_spot(centre_xy, frame_wh, spots):
    x, y = centre_xy[0] / frame_wh[0], centre_xy[1] / frame_wh[1]
    return any(x1 <= x <= x2 and y1 <= y <= y2 for (x1, y1, x2, y2) in spots)
