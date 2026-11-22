"""Dataset building and fine-tuning (transfer learning) for the classes COCO does not have.

COCO-pretrained YOLOv8 already knows person, sink, knife, bowl, carrot, broccoli, cell phone ...
Our own classes (cockroach, rat, fly, cap/hairnet, no_cap, gloves, spill, oil_used ...) are taught by
fine-tuning: start from the pretrained weights, freeze the backbone, train the head on a few hundred
labelled images, then evaluate on a held-out split -> precision, recall, mAP@0.5, mAP@0.5:0.95.
"""
from pathlib import Path
import cv2


def extract_frames(video_dir, out_dir, every_s=0.5, max_per_clip=400):
    """Pull a frame every `every_s` seconds from each clip, for labelling in Roboflow / CVAT."""
    video_dir, out_dir = Path(video_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = 0
    for v in sorted(video_dir.iterdir()):
        if v.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
            continue
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * every_s)))
        i = n = 0
        while n < max_per_clip:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                cv2.imwrite(str(out_dir / f"{v.stem}_{i:06d}.jpg"), frame)
                n += 1
            i += 1
        cap.release()
        n_total += n
    print(f"Saved {n_total} frames to {out_dir}")
    return n_total


def finetune(data_yaml, base="yolov8n.pt", epochs=60, imgsz=640, freeze=10, batch=16, project="runs", name="kitchen"):
    """Transfer learning: pretrained weights, first `freeze` layers frozen, early stopping on validation loss."""
    from ultralytics import YOLO
    model = YOLO(base)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, freeze=freeze, batch=batch, patience=15,
                project=project, name=name, exist_ok=True,
                hsv_v=0.5, fliplr=0.5, degrees=5, scale=0.4, mosaic=1.0)   # augmentation: brightness, flips, scale
    best = Path(project) / name / "weights" / "best.pt"
    return str(best)


def detection_metrics(weights, data_yaml, split="test"):
    """Precision, recall, mAP@0.5 and mAP@0.5:0.95 per class on a held-out split."""
    from ultralytics import YOLO
    m = YOLO(weights).val(data=str(data_yaml), split=split, verbose=False)
    names = m.names
    rows = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append(dict(cls=names[int(c)], precision=round(float(p), 3), recall=round(float(r), 3),
                         mAP50=round(float(ap50), 3), mAP50_95=round(float(ap), 3)))
    overall = dict(cls="ALL", precision=round(float(m.box.mp), 3), recall=round(float(m.box.mr), 3),
                   mAP50=round(float(m.box.map50), 3), mAP50_95=round(float(m.box.map), 3))
    return rows, overall
