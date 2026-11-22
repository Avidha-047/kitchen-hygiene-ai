"""Perception layer: video frames -> FrameObs (people with IDs and wrists, objects).

Algorithms
- People + 17 body keypoints: YOLOv8-pose (one-stage CNN detector with a keypoint head).
- Tracking: ByteTrack (Kalman filter motion model + Hungarian assignment on IoU), built into ultralytics.
- Objects: YOLOv8 pretrained on COCO (80 classes incl. sink, knife, bowl, carrot, broccoli, cell phone),
  plus any fine-tuned custom models (pests, PPE, spills ...).
"""
import cv2
import numpy as np
from .observations import PersonObs, ObjObs, FrameObs

L_WRIST, R_WRIST = 9, 10


def lather_score(image, person: PersonObs, min_conf=0.3):
    """Share of near-white, low-saturation pixels in a square around the wrists (soap foam is white)."""
    pts = person.visible_wrists(min_conf)
    if not pts or image is None:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    r = int(0.18 * person.height)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, int(cx - r)), max(0, int(cy - r)), min(w, int(cx + r)), min(h, int(cy + r))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 40) & (hsv[..., 2] > 200)
    return float(white.mean())


class Perceiver:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        self.YOLO = YOLO
        self.obj_model = YOLO(cfg.object_model)
        self.custom = [YOLO(p) for p in cfg.custom_models]
        self.reset()

    def reset(self):
        """New clip = fresh tracker state (re-loading the pose model resets ByteTrack)."""
        self.pose_model = self.YOLO(self.cfg.pose_model)

    def _objects(self, frame):
        objs = []
        for model in [self.obj_model] + self.custom:
            res = model.predict(frame, conf=self.cfg.object_conf, imgsz=self.cfg.imgsz, verbose=False)[0]
            if res.boxes is None:
                continue
            for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                name = model.names[int(k)]
                if model is self.obj_model and name == "person":
                    continue
                if model is self.obj_model and name == "mouse":   # COCO "mouse" is a computer mouse
                    continue
                objs.append(ObjObs(cls=name, conf=float(c), box=tuple(map(float, b))))
        return objs

    def __call__(self, frame, t, frame_idx):
        h, w = frame.shape[:2]
        res = self.pose_model.track(frame, persist=True, tracker=self.cfg.tracker, conf=self.cfg.person_conf,
                                    imgsz=self.cfg.imgsz, classes=[0], verbose=False)[0]
        persons = []
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            kxy = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
            kcf = res.keypoints.conf.cpu().numpy() if res.keypoints is not None and res.keypoints.conf is not None else None
            for i, tid in enumerate(ids):
                wrists = []
                for k in (L_WRIST, R_WRIST):
                    if kxy is not None:
                        x, y = kxy[i][k]
                        c = float(kcf[i][k]) if kcf is not None else 1.0
                        wrists.append((float(x), float(y), c if (x > 0 or y > 0) else 0.0))
                    else:
                        wrists.append((0.0, 0.0, 0.0))
                p = PersonObs(track_id=int(tid), box=tuple(map(float, boxes[i])), conf=float(confs[i]), wrists=wrists)
                p.lather = lather_score(frame, p, self.cfg.keypoint_conf)
                persons.append(p)
        return FrameObs(t=t, frame_idx=frame_idx, width=w, height=h, persons=persons, objects=self._objects(frame))


def iter_frames(path, sample_fps):
    """Yield (frame, t_seconds, frame_index) at roughly `sample_fps`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield frame, idx / fps, idx
        idx += 1
    cap.release()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0, frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    info["duration_s"] = info["frames"] / info["fps"] if info["fps"] else 0
    return info


def first_frame(path, at_s=1.0):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_s * fps))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
