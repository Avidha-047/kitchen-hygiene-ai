"""Kitchen Hygiene AI — web demo (Gradio).

Upload a clip from the kitchen camera (or pick a sample) and the system finds each person and their
wrists, checks the hygiene rules over time, and returns explained alerts with evidence.
"""
import glob
import html
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from kitchen_hygiene.calibrate import calibrate_pests
from kitchen_hygiene.config import Config
from kitchen_hygiene.perception import Perceiver
from kitchen_hygiene.pipeline import process_video

ROOT = Path(__file__).parent
SAMPLES = sorted(glob.glob(str(ROOT / "samples" / "*.mp4")))
MODEL = ROOT / "models" / "best.pt"

# Zones for the demo kitchen camera (fractions of the frame). A new camera needs these marked once.
ZONES = {
    "sink":       [[0.05, 0.38], [0.33, 0.38], [0.33, 0.90], [0.05, 0.90]],
    "soap":       [[0.15, 0.34], [0.24, 0.34], [0.24, 0.55], [0.15, 0.55]],
    "prep":       [[0.36, 0.52], [0.55, 0.52], [0.55, 0.75], [0.36, 0.75]],
    "storage":    [[0.45, 0.42], [0.56, 0.42], [0.56, 0.51], [0.45, 0.51]],
    "used_oil":   [[0.56, 0.46], [0.62, 0.46], [0.62, 0.64], [0.56, 0.64]],
    "cooking":    [[0.61, 0.54], [0.87, 0.54], [0.87, 0.93], [0.61, 0.93]],
    "clean_rack": [[0.67, 0.15], [0.96, 0.15], [0.96, 0.35], [0.67, 0.35]],
    "used_rack":  [[0.66, 0.36], [0.95, 0.36], [0.95, 0.58], [0.66, 0.58]],
}

cfg = Config()
cfg.zones = ZONES
cfg.sample_fps = 5
cfg.wash_min_s = 6                      # demo clips are short; the WHO figure is 20 s
cfg.custom_models = [str(MODEL)] if MODEL.exists() else []
cfg.thresholds = {"HANDWASH_VIOLATION": 0.935, "VEGETABLE_WASH_VIOLATION": 0.886, "PEST_DETECTED": 0.583}
if cfg.custom_models and len(SAMPLES) >= 2:
    cfg.pest_ignore_boxes, _ = calibrate_pests(SAMPLES, cfg)   # empty-kitchen calibration
PERCEIVER = Perceiver(cfg)                                       # load the models once

NAMES = {
    "HANDWASH_VIOLATION": "Hand washing",
    "VEGETABLE_WASH_VIOLATION": "Vegetable washing",
    "DIRTY_UTENSIL_VIOLATION": "Dirty utensil",
    "CROSS_CONTAMINATION_RISK": "Raw → cooked food",
    "PPE_VIOLATION": "Head cover / gloves",
    "PEST_DETECTED": "Pest",
    "SURFACE_NOT_CLEANED": "Surface cleaning",
    "OIL_REUSE_FLAG": "Reused oil",
}
SEV_COL = {"Critical": "#7A1F12", "High": "#C2441C", "Medium": "#B7791F", "Low": "#2F7D6B"}


def outcome(f):
    if f["status"] == "compliant":
        return "Compliant"
    if f["status"] == "violation" and f.get("reported"):
        return "Violation"
    return "Uncertain"


def summary_html(findings, seconds):
    v = sum(outcome(f) == "Violation" for f in findings)
    u = sum(outcome(f) == "Uncertain" for f in findings)
    c = sum(outcome(f) == "Compliant" for f in findings)
    status, col = ("VIOLATION FOUND", "#C2441C") if v else (("CHECK NEEDED", "#B7791F") if u else ("ALL CLEAR", "#2F7D6B"))
    card = lambda n, label, colr: (f'<div style="flex:1;min-width:110px;padding:14px;border-radius:12px;background:{colr}14;'
                                   f'border:1px solid {colr}55"><div style="font-size:28px;font-weight:700;color:{colr}">{n}</div>'
                                   f'<div style="font-size:13px;opacity:.8">{label}</div></div>')
    return (f'<div style="padding:14px 18px;border-radius:12px;background:{col};color:white;font-weight:700;'
            f'font-size:18px;letter-spacing:.5px;margin-bottom:10px">{status}'
            f'<span style="font-weight:400;font-size:14px;opacity:.9"> · {seconds:.0f} s of video checked</span></div>'
            f'<div style="display:flex;gap:10px;flex-wrap:wrap">{card(v, "Violations", "#C2441C")}'
            f'{card(u, "Uncertain / below threshold", "#B7791F")}{card(c, "Compliant checks", "#2F7D6B")}</div>')


def explain_md(findings):
    if not findings:
        return "_Nothing to check in this clip: nobody prepared food and no pest was seen._"
    out = []
    for f in findings:
        ev = f.get("evidence") or {}
        lines = [f"**{NAMES.get(f['vtype'], f['vtype'])} — {outcome(f)}** · confidence {f['confidence']:.2f} · "
                 f"{f['t_start']:.1f}–{f['t_end']:.1f} s", f"> {f['reason']}"]
        facts = []
        if "food_prep_start" in ev:
            facts.append(f"food prep started at {ev['food_prep_start']} s")
        if "visibility" in ev:
            facts.append(f"person visible {ev['visibility'] * 100:.0f}% of the time")
        if "washes_before" in ev:
            facts.append(f"washes before it: {len(ev['washes_before'])}")
        if "p_soap" in ev:
            facts.append(f"P(soap used) = {ev['p_soap']}")
        if "moved_frac_width" in ev:
            facts.append(f"moved {ev['moved_frac_width'] * 100:.0f}% of the frame width")
        if facts:
            lines.append("Evidence: " + "; ".join(facts) + ".")
        out.append("\n\n".join(lines))
    return "\n\n---\n\n".join(out)


def to_browser_mp4(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], check=True)
    return dst


def analyse(video, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a clip or pick a sample first.")
    work = Path(tempfile.mkdtemp())
    clip = work / (Path(video).stem.replace(" ", "_")[:40] + ".mp4")
    shutil.copy(video, clip)
    progress(0.1, desc="Finding people, wrists and objects…")
    r = process_video(clip, cfg, work / "out", perceiver=PERCEIVER, save_clips=False, progress=False)
    progress(0.85, desc="Preparing the evidence…")
    findings = r["findings"]
    rows = [{"Rule": NAMES.get(f["vtype"], f["vtype"]), "Result": outcome(f), "Severity": f.get("severity", ""),
             "Confidence": round(f["confidence"], 2), "When": f"{f['t_start']:.1f}–{f['t_end']:.1f} s",
             "Why": f["reason"]} for f in findings]
    table = pd.DataFrame(rows, columns=["Rule", "Result", "Severity", "Confidence", "When", "Why"])
    gallery = [(f["keyframe"], f"{NAMES.get(f['vtype'], f['vtype'])} · {outcome(f)} · {f['confidence']:.2f}")
               for f in findings if f.get("keyframe") and os.path.exists(f["keyframe"])]
    annotated = to_browser_mp4(r["annotated"], str(work / "annotated.mp4")) if r.get("annotated") else None
    return summary_html(findings, r["info"]["duration_s"]), table, gallery, annotated, explain_md(findings)


INTRO = """
# 🍳 Kitchen Hygiene AI
**Can AI be a kitchen hygiene inspector?** Upload a clip from the kitchen camera, or pick a sample. The system finds every person
and their wrists, follows them through the clip, checks hygiene rules like *"wash your hands before preparing food"*, and explains
every alert with the evidence behind it. When it cannot see enough, it says **uncertain** instead of blaming anyone.
"""

HOW = """
### How it works
| Step | What happens | Algorithm |
|---|---|---|
| See | Find people and their 17 body points (we use the wrists) | **YOLOv8-pose** (CNN) |
| Follow | Keep the same ID for each person across frames | **ByteTrack** (Kalman filter + Hungarian matching) |
| Spot | Everyday objects, plus our own **cockroach detector** | **YOLOv8** (COCO) + **YOLOv8n fine-tuned** on 310 photos |
| Understand | Washing = both wrists at the sink, together and moving 3+ s; food prep = wrists at the board 2+ s | Rules over keypoints and zones |
| Decide | Was soap used? Dispenser touched, lather seen, rub long enough | **Bayesian network** |
| Judge | *For every person p: food prep with no earlier wash ⇒ violation* | **First-order rules**, one memory per person |
| Prove | Severity, confidence, evidence frame, annotated video | Evidence log + dashboard |

**Our cockroach detector** (67 unseen test photos): precision 0.82 · recall 0.81 · mAP@0.5 0.87.
Pest look-alikes fixed in this camera's view (a stove knob, a drain) are found once by calibrating on the empty kitchen and ignored.

### Honest notes
The demo kitchen and all sample clips are **AI-generated** (Google Veo) from one fixed CCTV view; real kitchen footage is the next step.
On our 12 test scenes the system scored F1 0.50 overall; hand-washing was the strongest rule (every hand-wash alert was correct).
Zones are set for this demo camera, so clips from other cameras will not be judged correctly until their zones are marked.
No face recognition: people are anonymous IDs.

Built by **Avidha Chandra** · [Code on GitHub](https://github.com/Avidha-047/kitchen-hygiene-ai)
"""

with gr.Blocks(title="Kitchen Hygiene AI") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Inspect a clip"):
        with gr.Row():
            with gr.Column(scale=4):
                video_in = gr.Video(label="Kitchen camera clip", sources=["upload"], height=300)
                if SAMPLES:
                    gr.Examples(SAMPLES, inputs=video_in, label="Sample clips (click one)")
                run = gr.Button("Inspect this clip", variant="primary", size="lg")
                gr.Markdown("<small>About a minute per 10-second clip on the free CPU.</small>")
            with gr.Column(scale=6):
                summary = gr.HTML()
                table = gr.Dataframe(value=pd.DataFrame(columns=["Rule", "Result", "Severity", "Confidence", "When", "Why"]), label="Findings", wrap=True, interactive=False)
        with gr.Row():
            gallery = gr.Gallery(label="Evidence frames", columns=2, height=330, object_fit="contain")
            video_out = gr.Video(label="What the system saw: zones, person IDs, wrists, alerts", height=330)
        with gr.Accordion("Why did it decide that?", open=False):
            why = gr.Markdown()
        run.click(analyse, inputs=video_in, outputs=[summary, table, gallery, video_out, why])
    with gr.Tab("How it works"):
        gr.Markdown(HOW)
        arch = next((p for p in [ROOT / "docs" / "architecture.png", ROOT / "docs" / "images" / "architecture.png"] if p.exists()), None)
        if arch:
            gr.Image(str(arch), show_label=False)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(theme=gr.themes.Soft(primary_hue="teal"))
