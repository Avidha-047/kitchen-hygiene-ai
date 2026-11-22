"""End-to-end pipeline: video -> perception -> temporal/rule engine -> evidence -> database.

    Live video -> frame sampling -> detection + tracking + keypoints -> zone/action events
    -> temporal reasoning -> hygiene rules -> violation + severity -> evidence (frame, clip) -> dashboard
"""
import json
import shutil
import sqlite3
import subprocess
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import cv2

from .engine import HygieneEngine
from .geometry import Zones, draw_obs, draw_zones
from .perception import Perceiver, iter_frames, video_info

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp"}


def _to_h264(src: Path):
    """Re-encode to H.264 so browsers (and Colab) can play the clip. Keeps the original if ffmpeg is missing."""
    if not shutil.which("ffmpeg"):
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], capture_output=True)
    if r.returncode == 0 and dst.exists():
        src.unlink(missing_ok=True)
        dst.rename(src)
    return src


def _cut_clip(video, t0, t1, out_path, zones_norm, banner):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zones = Zones(zones_norm, w, h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, t0) * fps))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = int((t1 - max(0, t0)) * fps)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t = max(0, t0) + i / fps
        frame = draw_zones(frame, zones, alpha=0.15)
        cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 20), -1)
        cv2.putText(frame, f"{banner}   t={t:5.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 120, 255), 2)
        writer.write(frame)
    writer.release()
    cap.release()
    return _to_h264(Path(out_path))


class Store:
    """SQLite store for findings and events (the evidence log the manager's report reads from)."""

    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("""CREATE TABLE IF NOT EXISTS findings(
            id INTEGER PRIMARY KEY, clip TEXT, vtype TEXT, detail TEXT, status TEXT, reported INTEGER,
            severity TEXT, person INTEGER, t_start REAL, t_end REAL, confidence REAL, reason TEXT,
            evidence TEXT, keyframe TEXT, clip_path TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY, clip TEXT, etype TEXT, person INTEGER, t_start REAL, t_end REAL, info TEXT)""")
        self.db.commit()

    def clear_clip(self, clip):
        self.db.execute("DELETE FROM findings WHERE clip=?", (clip,))
        self.db.execute("DELETE FROM events WHERE clip=?", (clip,))

    def add(self, clip, findings, events):
        for f in findings:
            self.db.execute("INSERT INTO findings(clip,vtype,detail,status,reported,severity,person,t_start,t_end,confidence,reason,evidence,keyframe,clip_path) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (clip, f["vtype"], f["detail"], f["status"], int(f["reported"]), f["severity"], f["person"],
                             f["t_start"], f["t_end"], f["confidence"], f["reason"], json.dumps(f["evidence"]),
                             f.get("keyframe"), f.get("clip_path")))
        for e in events:
            self.db.execute("INSERT INTO events(clip,etype,person,t_start,t_end,info) VALUES(?,?,?,?,?,?)",
                            (clip, e.etype, e.person, e.t_start, e.t_end, json.dumps(e.info)))
        self.db.commit()


def process_video(video, cfg, out_dir, perceiver=None, annotate=True, save_clips=True, progress=True):
    """Run the whole system on one clip. Returns a dict with findings, events and file paths."""
    video, out_dir = Path(video), Path(out_dir)
    clip = video.stem
    ev_dir = out_dir / "evidence" / clip
    ev_dir.mkdir(parents=True, exist_ok=True)
    perceiver = perceiver or Perceiver(cfg)
    perceiver.reset()
    info = video_info(video)
    classes = set(perceiver.obj_model.names.values())
    for m in perceiver.custom:
        classes |= set(m.names.values())
    engine = HygieneEngine(cfg, info["width"], info["height"], available_classes=classes)

    ring = OrderedDict()                      # frame_idx -> (image, FrameObs) for keyframes
    ring_len = int(cfg.sample_fps * 20)
    writer = None
    if annotate:
        writer = cv2.VideoWriter(str(ev_dir / f"{clip}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 cfg.sample_fps, (info["width"], info["height"]))
    keyframes = {}
    alerts_on_screen = []
    total = max(1, int(info["duration_s"] * cfg.sample_fps))

    def keyframe_for(f):
        idx = f.frame_idx
        if idx not in ring and ring:
            idx = min(ring, key=lambda k: abs(k - idx))
        if idx not in ring:
            return None
        img, fo = ring[idx]
        label = f"{f.vtype}{':' + f.detail if f.detail else ''} [{f.status}] P{f.person if f.person is not None else '-'} conf {f.confidence:.2f}"
        out = draw_obs(img, fo, engine.zones, highlight_id=f.person, banner=label)
        return out

    def handle(new, fo_now):
        for f in new:
            key = id(f)
            kf = keyframe_for(f)
            if kf is not None:
                p = ev_dir / f"{clip}_{len(keyframes):03d}_{f.vtype.lower()}.jpg"
                cv2.imwrite(str(p), kf)
                keyframes[key] = str(p)
            if f.status != "compliant":
                alerts_on_screen.append((fo_now.t + 4.0, f"{f.vtype} P{f.person if f.person is not None else '-'} ({f.status})"))

    for i, (frame, t, idx) in enumerate(iter_frames(video, cfg.sample_fps)):
        fo = perceiver(frame, t, idx)
        ring[idx] = (frame, fo)
        while len(ring) > ring_len:
            ring.popitem(last=False)
        handle(engine.update(fo), fo)
        if writer is not None:
            alerts_on_screen[:] = [a for a in alerts_on_screen if a[0] >= t]
            banner = " | ".join(a[1] for a in alerts_on_screen[-2:]) or f"monitoring  t={t:5.1f}s"
            writer.write(draw_obs(frame, fo, engine.zones, banner=banner))
        if progress and i % 50 == 0:
            print(f"  {clip}: {100 * i / total:5.1f}%", end="\r")
    if ring:
        last_fo = list(ring.values())[-1][1]
        handle(engine.finish(), last_fo)
    if writer is not None:
        writer.release()
        _to_h264(ev_dir / f"{clip}_annotated.mp4")

    findings = []
    for f in engine.findings:
        d = asdict(f)
        d["severity"] = f.severity
        d["reported"] = f.status == "violation" and f.confidence >= cfg.threshold(f.vtype)
        d["keyframe"] = keyframes.get(id(f))
        d["clip"] = clip
        if save_clips and f.status != "compliant":
            t0 = f.t_start - cfg.clip_pad_s
            t1 = f.t_end + cfg.clip_pad_s
            if t1 - t0 > 20:
                t0 = t1 - 20
            p = ev_dir / f"{clip}_{len(findings):03d}_{f.vtype.lower()}.mp4"
            d["clip_path"] = str(_cut_clip(video, t0, t1, p, cfg.zones, f"{f.vtype} P{f.person}"))
        findings.append(d)

    store = Store(out_dir / "hygiene.db")
    store.clear_clip(clip)
    store.add(clip, findings, engine.events)
    with open(ev_dir / f"{clip}_findings.json", "w") as fh:
        json.dump({"clip": clip, "info": info, "findings": findings,
                   "events": [asdict(e) for e in engine.events]}, fh, indent=2, default=str)
    if progress:
        n = sum(1 for f in findings if f["reported"])
        print(f"  {clip}: done, {n} violation(s) reported, {len(findings)} finding(s) in total")
    return {"clip": clip, "findings": findings, "events": engine.events, "info": info,
            "annotated": str(ev_dir / f"{clip}_annotated.mp4") if annotate else None}


def process_folder(folder, cfg, out_dir, **kw):
    folder = Path(folder)
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT)
    perceiver = Perceiver(cfg)
    results = []
    for v in videos:
        print(f"Processing {v.name}")
        results.append(process_video(v, cfg, out_dir, perceiver=perceiver, **kw))
    return results
