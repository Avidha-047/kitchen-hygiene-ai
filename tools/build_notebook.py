"""Builds notebooks/Kitchen_Hygiene_Monitor.ipynb. Run from the repo root:  python tools/build_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Can AI Be a Kitchen Hygiene Inspector? — prototype

Live kitchen video → **detected hygiene events + violations + evidence**.

| Layer | Algorithm | Library |
|---|---|---|
| People + wrist keypoints | YOLOv8-pose (one-stage CNN detector with a keypoint head) | `ultralytics` |
| Tracking (one ID per person) | ByteTrack (Kalman filter + Hungarian assignment) | `ultralytics` |
| Objects (sink, knife, carrot, phone …) | YOLOv8, pretrained on COCO | `ultralytics` |
| Our own classes (pests, cap, spill …) | YOLOv8 fine-tuned on our photos (transfer learning) | `ultralytics` |
| Actions (hand-rubbing, food prep, zone visits) | Rule-based temporal recognition on keypoints + zones | this repo |
| "Was soap used?" | Bayesian network (3 evidence nodes) | this repo |
| Decision | First-order hygiene rules, forward chaining | this repo |
| Evaluation | Precision / recall / F1, mAP, confusion matrix, PR-curve threshold tuning | this repo + `ultralytics` |

**Before you start:** Runtime → Change runtime type → **T4 GPU**. Then run the cells top to bottom.
""")

md("## 1. Setup")
code("""
!pip install -q ultralytics lap
from google.colab import drive
drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/KitchenHygiene'     # <- your shared Drive folder
CODE = f'{BASE}/code'                              # the unzipped project folder (contains kitchen_hygiene/)
VIDEOS = f'{BASE}/videos'                          # S01.mp4, S02.mp4 ... H6.mp4
OUT = f'{BASE}/output'
GT = f'{BASE}/ground_truth.csv'

import sys, os, json, glob
sys.path.insert(0, CODE)
os.makedirs(OUT, exist_ok=True)
from kitchen_hygiene.config import Config
print('clips found:', sorted(os.path.basename(p) for p in glob.glob(f'{VIDEOS}/*')))
""")

md("""**AI-generated clips come in short parts** (S01_a.mp4, S01_b.mp4 ...). This cell joins the parts of each scene
into one clip (S01.mp4) and moves the parts into `videos/parts/`. Safe to run again.""")
code("""
!python "$CODE/tools/join_parts.py" "$VIDEOS"
""")

md("""## 2. Check the reasoning layer (18 scripted scenarios, no video needed)
These tests feed hand-written observations into the rule engine: wash with and without soap, two cooks, occlusion,
re-linking a lost track, phone after washing, vegetables, utensils, raw → cooked, pests, spills, PPE.""")
code("""
!cd "$CODE" && python tests/test_engine.py
""")

md("""## 3. Mark the zones (once per camera position)
Look at the frame with the 10% grid, then edit the polygons below: each point is `[x, y]` as a fraction of the
width and height. Required: `sink`, `prep`. Add the optional ones your footage has.""")
code("""
import cv2, matplotlib.pyplot as plt
from kitchen_hygiene.perception import first_frame
from kitchen_hygiene.geometry import draw_grid, draw_zones, Zones

sample = sorted(glob.glob(f'{VIDEOS}/*'))[0]
frame = first_frame(sample, at_s=1.0)
plt.figure(figsize=(14, 8)); plt.imshow(cv2.cvtColor(draw_grid(frame), cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
""")
code("""
ZONES = {   # marked once on the base frame of our AI-generated kitchen (fixed CCTV view)
    "sink":       [[0.05,0.38],[0.33,0.38],[0.33,0.90],[0.05,0.90]],
    "soap":       [[0.15,0.34],[0.24,0.34],[0.24,0.55],[0.15,0.55]],
    "prep":       [[0.36,0.52],[0.55,0.52],[0.55,0.75],[0.36,0.75]],   # cutting board
    "storage":    [[0.45,0.42],[0.56,0.42],[0.56,0.51],[0.45,0.51]],   # vegetable basket
    "used_oil":   [[0.56,0.46],[0.62,0.46],[0.62,0.64],[0.56,0.64]],   # bottle marked USED
    "cooking":    [[0.61,0.54],[0.87,0.54],[0.87,0.93],[0.61,0.93]],   # stove
    "clean_rack": [[0.67,0.15],[0.96,0.15],[0.96,0.35],[0.67,0.35]],
    "used_rack":  [[0.66,0.36],[0.95,0.36],[0.95,0.58],[0.66,0.58]],
    # "raw" and "cooked" zones go where the raw chicken and the rice bowl sit in your own footage
}
z = Zones(ZONES, frame.shape[1], frame.shape[0])
plt.figure(figsize=(14, 8)); plt.imshow(cv2.cvtColor(draw_zones(frame, z, 0.3), cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
""")

md("""## 4. (Optional) Teach the model our own classes — transfer learning
COCO already knows person, sink, knife, bowl, carrot, broccoli and cell phone. For **cockroach, rat, fly, cap,
no_cap, gloves, spill, oil_used** we fine-tune on our labelled photos.

1. Run the first cell to pull frames from the clips into `frames_for_labelling/`.
2. Label them (with the photos from the shot list) in Roboflow or CVAT; export **YOLOv8 format** with train/valid/test splits
   into `BASE/dataset/` (it contains `data.yaml`).
3. Run the training cell. It starts from pretrained weights, freezes the first 10 layers and stops early when
   validation stops improving. The last cell reports precision, recall and mAP on the **test** split.""")
md("""**Option A: start today with public data.** This downloads 310 labelled cockroach photos from a public GitHub
repository (nafiz5662/cockroach-detection, Pascal VOC boxes) and converts them to YOLO format, split into
train / valid / test. Train on this first, then add your own kitchen photos as more classes.""")
code("""
!git clone --depth 1 --filter=blob:none --sparse https://github.com/nafiz5662/cockroach-detection /content/roach_src
!cd /content/roach_src && git sparse-checkout set --no-cone research/object_detection/images
!python -I "$CODE/tools/voc_to_yolo.py" /content/roach_src/research/object_detection/images/train /content/roach_src/research/object_detection/images/test "$BASE/dataset"
""")
code("""
from kitchen_hygiene.train import extract_frames, finetune, detection_metrics
extract_frames(VIDEOS, f'{BASE}/frames_for_labelling', every_s=0.5)
""")
code("""
DATA = f'{BASE}/dataset/data.yaml'
CUSTOM = finetune(DATA, base='yolov8n.pt', epochs=60, imgsz=416, project=f'{BASE}/runs', name='kitchen416')
rows, overall = detection_metrics(CUSTOM, DATA, split='test')
import pandas as pd
display(pd.DataFrame(rows + [overall]))
""")
md("""**Proof of training (put these in the slides and the report).** Ultralytics saves the loss and mAP curves per epoch,
the confusion matrix and the precision-recall curve in the run folder.""")
code("""
from IPython.display import Image, display
run = f'{BASE}/runs/kitchen416'
for name in ['results.png', 'confusion_matrix_normalized.png', 'confusion_matrix.png', 'BoxPR_curve.png', 'PR_curve.png']:
    p = os.path.join(run, name)
    if os.path.exists(p):
        print(name); display(Image(p, width=900))
""")

md("## 5. Run the system on every clip")
code("""
cfg = Config()
cfg.zones = ZONES
cfg.sample_fps = 5
cfg.wash_min_s = 20          # WHO: at least 20 s of rubbing
cfg.wash_min_s = 6           # AI clips are 5-8 s each, so a 20 s wash cannot be shown; we scale it down (say so in the report)
custom = f'{BASE}/runs/kitchen416/weights/best.pt'
cfg.custom_models = [custom] if os.path.exists(custom) else []
print('custom model:', cfg.custom_models or 'none (pests / PPE / spill rules need it)')

# Empty-kitchen calibration: spots the pest detector "finds" in the empty starting frame of 2+ clips are look-alikes
from kitchen_hygiene.calibrate import calibrate_pests
cfg.pest_ignore_boxes, calib = calibrate_pests(sorted(glob.glob(f'{VIDEOS}/*.mp4')), cfg)
print('look-alike spots ignored by the pest detector:', len(cfg.pest_ignore_boxes))
if calib is not None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 5)); plt.imshow(calib[..., ::-1]); plt.axis('off'); plt.title('Empty-kitchen calibration'); plt.show()
cfg.save(f'{OUT}/config.json')

from kitchen_hygiene.pipeline import process_folder
results = process_folder(VIDEOS, cfg, OUT)
findings = [f for r in results for f in r['findings']]
video_hours = sum(r['info']['duration_s'] for r in results) / 3600
""")
code("""
import pandas as pd
cols = ['clip', 'vtype', 'detail', 'status', 'reported', 'severity', 'person', 't_start', 't_end', 'confidence', 'reason']
df = pd.DataFrame(findings, columns=cols)
print(len(df), 'findings;', int(df['reported'].sum()) if len(df) else 0, 'reported as violations')
df.sort_values(['clip', 't_start'])
""")

md("## 6. Watch an annotated clip (for the demo)")
code("""
from IPython.display import Video
clip = 'S02'      # change to any clip name
Video(f'{OUT}/evidence/{clip}/{clip}_annotated.mp4', embed=True, width=800)
""")

md("## 7. The manager's dashboard / weekly evidence report")
code("""
from kitchen_hygiene.dashboard import build_dashboard
from IPython.display import HTML
path = build_dashboard(findings, f'{OUT}/dashboard.html', period='Scenario test week')
HTML(open(path).read())
""")

md("""## 8. Evaluation against the answer key
`ground_truth.csv` is the log sheet from filming. Clips are split in two: thresholds are **tuned on the validation half**
(precision–recall sweep, best F1) and results are **reported on the test half**, so the numbers are not inflated.""")
code("""
from kitchen_hygiene import evaluate as ev
gt = ev.load_ground_truth(GT)
clips = sorted({f['clip'] for f in findings} | {g['clip'] for g in gt})
val, test = ev.split_clips(clips, val_frac=0.5)
f_val = [f for f in findings if f['clip'] in val]; g_val = [g for g in gt if g['clip'] in val]
f_test = [f for f in findings if f['clip'] in test]; g_test = [g for g in gt if g['clip'] in test]

best, curves = ev.tune_thresholds(f_val, g_val)
print('thresholds chosen on validation clips:', best)

test_hours = sum(r['info']['duration_s'] for r in results if r['clip'] in test) / 3600
rows, overall, detail = ev.report(f_test, g_test, test_hours, thresholds=best)
display(pd.DataFrame(rows + [overall]))
""")
code("""
# Precision-recall curves from the validation sweep
if not curves:
    print('No violations of a labelled type were predicted on the validation clips yet.')
for v, pts in curves.items():
    plt.plot([p['recall'] for p in pts], [p['precision'] for p in pts], marker='o', label=v)
if curves:
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Threshold sweep (validation clips)'); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.show()
""")
code("""
labels, m = ev.confusion(findings, gt, thresholds=best)
display(pd.DataFrame(m).T.reindex(columns=labels).fillna(0).astype(int))
""")
code("""
# The same report on ALL clips (thresholds from validation), plus which rule is weakest
rows_all, overall_all, detail_all = ev.report(findings, gt, video_hours, thresholds=best)
display(pd.DataFrame(rows_all + [overall_all]))
weakest = min(rows_all, key=lambda r: r['F1']) if rows_all else None
print('Weakest rule:', weakest)
""")

md("## 9. Error analysis: every false alarm and every miss, with its evidence frame")
code("""
from IPython.display import Image, display
print('FALSE POSITIVES')
for f in detail_all['fp']:
    print(f"{f['clip']} {f['t_start']:.1f}-{f['t_end']:.1f}s {f['vtype']} P{f['person']} conf {f['confidence']:.2f}: {f['reason']}")
    if f.get('keyframe'): display(Image(f['keyframe'], width=480))
print('\\nMISSED VIOLATIONS')
for g in detail_all['fn']:
    near = [f for f in findings if f['clip'] == g['clip'] and f['vtype'] == g['vtype']]
    print(f"{g['clip']} {g['start']}-{g['end']}s {g['vtype']} ({g['note']}) -> system said: {[(f['status'], f['confidence']) for f in near] or 'nothing'}")
""")

md("## 10. The soap Bayesian network, worked through")
code("""
from kitchen_hygiene.bayes import p_soap
rows = []
for d in (True, False, None):
    for l in (True, False, None):
        for r in (True, False):
            rows.append(dict(dispenser=d, lather=l, rubbed_20s=r, P_soap=round(p_soap(cfg, d, l, r), 3)))
pd.DataFrame(rows)
""")

nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "notebooks/Kitchen_Hygiene_Monitor.ipynb")
print("written")
