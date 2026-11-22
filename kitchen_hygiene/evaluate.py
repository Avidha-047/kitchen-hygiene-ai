"""Evaluation against the hand-written answer key (ground_truth.csv).

Event level (deliverable 20): did the system flag each scripted violation, within a few
seconds of when it happened, for the right rule? -> TP / FP / FN, precision, recall, F1 per rule,
a confusion matrix, missed violations and false alarms per hour.

Threshold tuning (sir's question "how do you find 0.6?"): sweep the confidence threshold on a
validation split and keep the value with the best F1 (ties -> higher precision).
"""
import csv
from collections import defaultdict

GT_MAP = {
    "HANDWASH_COMPLIANT": ("HANDWASH_VIOLATION", "compliant"),
    "VEGETABLE_WASH_COMPLIANT": ("VEGETABLE_WASH_VIOLATION", "compliant"),
    "DIRTY_UTENSIL_COMPLIANT": ("DIRTY_UTENSIL_VIOLATION", "compliant"),
    "CROSS_CONTAMINATION_COMPLIANT": ("CROSS_CONTAMINATION_RISK", "compliant"),
    "PPE_COMPLIANT": ("PPE_VIOLATION", "compliant"),
    "SURFACE_CLEANED": ("SURFACE_NOT_CLEANED", "compliant"),
}
VIOLATION_TYPES = ["HANDWASH_VIOLATION", "VEGETABLE_WASH_VIOLATION", "DIRTY_UTENSIL_VIOLATION",
                   "CROSS_CONTAMINATION_RISK", "PPE_VIOLATION", "PEST_DETECTED", "SURFACE_NOT_CLEANED", "OIL_REUSE_FLAG"]


def load_ground_truth(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            label = (r.get("event") or "").strip().upper()
            clip = (r.get("clip") or "").strip()
            clip = clip.rsplit(".", 1)[0] if "." in clip else clip
            if not clip or not label:
                continue
            if label == "NONE":
                vtype, status = None, "none"
            elif label in GT_MAP:
                vtype, status = GT_MAP[label]
            else:
                vtype, status = label, "violation"
            rows.append(dict(clip=clip, vtype=vtype, status=status,
                             start=float(r["start_s"]) if r.get("start_s") else 0.0,
                             end=float(r["end_s"]) if r.get("end_s") else 1e9,
                             person=(r.get("person") or "").strip(), note=r.get("note", "")))
    return rows


def _overlap(a0, a1, b0, b1, tol):
    return a0 - tol <= b1 and b0 - tol <= a1


def match(findings, gt, tol=3.0, thresholds=None, default_thr=0.5):
    """Greedy one-to-one matching of reported violations to ground-truth violations."""
    thresholds = thresholds or {}
    preds = [f for f in findings if f["status"] == "violation"
             and f["confidence"] >= thresholds.get(f["vtype"], default_thr)]
    gts = [g for g in gt if g["status"] == "violation"]
    used = set()
    tp, fp = [], []
    for p in sorted(preds, key=lambda x: -x["confidence"]):
        cand = [i for i, g in enumerate(gts) if i not in used and g["clip"] == p["clip"] and g["vtype"] == p["vtype"]
                and _overlap(p["t_start"], p["t_end"], g["start"], g["end"], tol)]
        if cand:
            used.add(cand[0])
            tp.append((p, gts[cand[0]]))
        else:
            fp.append(p)
    fn = [g for i, g in enumerate(gts) if i not in used]
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def report(findings, gt, video_hours, tol=3.0, thresholds=None, default_thr=0.5):
    tp, fp, fn = match(findings, gt, tol, thresholds, default_thr)
    per = defaultdict(lambda: [0, 0, 0])
    for p, g in tp:
        per[p["vtype"]][0] += 1
    for p in fp:
        per[p["vtype"]][1] += 1
    for g in fn:
        per[g["vtype"]][2] += 1
    rows = []
    for v in VIOLATION_TYPES:
        if v not in per:
            continue
        a, b, c = per[v]
        P, R, F = prf(a, b, c)
        rows.append(dict(rule=v, TP=a, FP=b, FN=c, precision=round(P, 2), recall=round(R, 2), F1=round(F, 2)))
    A, B, C = len(tp), len(fp), len(fn)
    P, R, F = prf(A, B, C)
    overall = dict(rule="ALL", TP=A, FP=B, FN=C, precision=round(P, 2), recall=round(R, 2), F1=round(F, 2),
                   false_alarms_per_hour=round(B / video_hours, 1) if video_hours else None,
                   missed_violations=C)
    return rows, overall, dict(tp=tp, fp=fp, fn=fn)


def confusion(findings, gt, tol=3.0, thresholds=None, default_thr=0.5):
    """Rows: what really happened in each labelled scene. Columns: what the system said for that rule."""
    thresholds = thresholds or {}
    labels = ["violation reported", "uncertain", "compliant / below threshold", "nothing said"]
    m = defaultdict(lambda: dict.fromkeys(labels, 0))
    for g in gt:
        if g["vtype"] is None:
            said = [f for f in findings if f["clip"] == g["clip"] and f["status"] == "violation"
                    and f["confidence"] >= thresholds.get(f["vtype"], default_thr)]
            m["no event (look-alike etc.)"]["violation reported" if said else "nothing said"] += 1
            continue
        fs = [f for f in findings if f["clip"] == g["clip"] and f["vtype"] == g["vtype"]
              and _overlap(f["t_start"], f["t_end"], g["start"], g["end"], tol)]
        if any(f["status"] == "violation" and f["confidence"] >= thresholds.get(f["vtype"], default_thr) for f in fs):
            col = "violation reported"
        elif any(f["status"] == "uncertain" for f in fs):
            col = "uncertain"
        elif fs:
            col = "compliant / below threshold"
        else:
            col = "nothing said"
        m[f"{g['status']}"][col] += 1
    return labels, dict(m)


def tune_thresholds(findings, gt, tol=3.0):
    """Precision-recall sweep per rule on a validation split. Returns best thresholds and the curves."""
    best, curves = {}, {}
    for v in VIOLATION_TYPES:
        cands = sorted({round(f["confidence"], 3) for f in findings if f["vtype"] == v and f["status"] == "violation"})
        if not cands or not any(g["vtype"] == v and g["status"] == "violation" for g in gt):
            continue
        pts = []
        for thr in cands:
            tp, fp, fn = match([f for f in findings if f["vtype"] == v], [g for g in gt if g["vtype"] == v], tol, {v: thr})
            P, R, F = prf(len(tp), len(fp), len(fn))
            pts.append(dict(threshold=thr, precision=round(P, 3), recall=round(R, 3), F1=round(F, 3)))
        curves[v] = pts
        top = max(pts, key=lambda d: (d["F1"], d["precision"], -d["threshold"]))
        best[v] = top["threshold"]
    return best, curves


def split_clips(clips, val_frac=0.5, seed=7):
    """Deterministic validation / test split of clip names (tune on one half, report on the other)."""
    import random
    c = sorted(clips)
    random.Random(seed).shuffle(c)
    k = max(1, int(len(c) * val_frac))
    return set(c[:k]), set(c[k:])
