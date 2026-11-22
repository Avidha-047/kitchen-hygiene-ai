"""Manager's dashboard / weekly evidence report as one self-contained HTML file."""
import base64
import html
import os
from collections import Counter
from datetime import datetime

import cv2

from .config import SEVERITY_RANK

SEV_COL = {"Critical": "#8E1B10", "High": "#C2441C", "Medium": "#B7791F", "Low": "#2F6F8F"}


def _thumb(path, width=240):
    if not path or not os.path.exists(path):
        return ""
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buf).decode()}" alt="evidence frame">' if ok else ""


def _bar_svg(counts):
    if not counts:
        return "<p class='muted'>No violations.</p>"
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    mx = max(v for _, v in items)
    rowh, w = 30, 520
    parts = [f'<svg viewBox="0 0 {w + 260} {rowh * len(items) + 10}" width="100%" style="max-width:820px" role="img" aria-label="violations by rule">']
    for i, (k, v) in enumerate(items):
        y = i * rowh + 5
        bw = max(4, int(w * v / mx))
        parts.append(f'<text x="250" y="{y + 19}" text-anchor="end" font-size="13" fill="#1F2A26">{html.escape(k)}</text>')
        parts.append(f'<rect x="260" y="{y + 4}" width="{bw}" height="20" rx="3" fill="#2F7D6B"/>')
        parts.append(f'<text x="{266 + bw}" y="{y + 19}" font-size="13" fill="#1F2A26">{v}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_dashboard(findings, out_path, title="Kitchen Hygiene Monitor", period="This week", rel_to=None):
    reported = [f for f in findings if f.get("reported")]
    uncertain = [f for f in findings if f["status"] == "uncertain" or (f["status"] == "violation" and not f.get("reported"))]
    compliant = [f for f in findings if f["status"] == "compliant"]
    worst = max((SEVERITY_RANK[f["severity"]] for f in reported), default=0)
    status = "VIOLATION" if worst >= 3 else ("WARNING" if worst else "COMPLIANT")
    status_col = "#C2441C" if worst >= 3 else ("#B7791F" if worst else "#2F7D6B")
    by_type = Counter((f["vtype"] + (":" + f["detail"] if f.get("detail") else "")) for f in reported)
    by_sev = Counter(f["severity"] for f in reported)
    base = os.path.dirname(os.path.abspath(out_path)) if rel_to is None else rel_to

    def row(f):
        clip_link = ""
        if f.get("clip_path") and os.path.exists(f["clip_path"]):
            rel = os.path.relpath(f["clip_path"], base)
            clip_link = f'<a href="{html.escape(rel)}">clip</a>'
        person = f"P{f['person']:02d}" if f.get("person") is not None else "—"
        sev = f["severity"]
        return (f"<tr><td>{html.escape(f['clip'])}<br><span class='muted'>{f['t_start']:.1f}–{f['t_end']:.1f}s</span></td>"
                f"<td><b>{html.escape(f['vtype'])}</b>{('<br><span class=muted>' + html.escape(f['detail']) + '</span>') if f.get('detail') else ''}</td>"
                f"<td><span class='sev' style='background:{SEV_COL.get(sev, '#555')}'>{sev}</span></td>"
                f"<td>{person}</td><td>{f['confidence']:.2f}</td><td>{html.escape(f['reason'])}</td>"
                f"<td>{_thumb(f.get('keyframe'))}<br>{clip_link}</td></tr>")

    reported_sorted = sorted(reported, key=lambda f: (-SEVERITY_RANK[f["severity"]], f["clip"], f["t_start"]))
    review_sorted = sorted(uncertain, key=lambda f: (f["clip"], f["t_start"]))
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:#F4F1EA;color:#1F2A26;margin:0;padding:28px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:28px 0 10px}}
.muted{{color:#5B6762;font-size:13px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-top:18px}}
.card{{background:#FBF9F4;border:1px solid #D9D2C2;border-radius:12px;padding:14px 18px;min-width:150px}}
.card .n{{font-size:30px;font-weight:700}}
.status{{display:inline-block;color:#fff;border-radius:8px;padding:6px 14px;font-weight:700;letter-spacing:.05em}}
table{{border-collapse:collapse;width:100%;background:#FBF9F4;font-size:14px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #E2DCCD;vertical-align:top}}
th{{background:#1F2A26;color:#F4F1EA}}
td img{{border-radius:6px;max-width:240px}}
.sev{{color:#fff;border-radius:6px;padding:2px 8px;font-size:12px;font-weight:600}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="muted">{html.escape(period)} · generated {now}</div>
<div class="cards">
 <div class="card"><div class="muted">Current status</div><div class="status" style="background:{status_col}">{status}</div></div>
 <div class="card"><div class="muted">Violations reported</div><div class="n">{len(reported)}</div></div>
 <div class="card"><div class="muted">Critical / High</div><div class="n">{by_sev.get('Critical', 0)} / {by_sev.get('High', 0)}</div></div>
 <div class="card"><div class="muted">Medium / Low</div><div class="n">{by_sev.get('Medium', 0)} / {by_sev.get('Low', 0)}</div></div>
 <div class="card"><div class="muted">To review (uncertain)</div><div class="n">{len(review_sorted)}</div></div>
 <div class="card"><div class="muted">Checks passed</div><div class="n">{len(compliant)}</div></div>
</div>
<h2>Violations by rule</h2>{_bar_svg(by_type)}
<h2>Violation log with evidence</h2>
<table><tr><th>Clip / time</th><th>Rule</th><th>Severity</th><th>Person</th><th>Conf.</th><th>Why it was flagged</th><th>Evidence</th></tr>
{''.join(row(f) for f in reported_sorted) or '<tr><td colspan=7>No violations reported.</td></tr>'}
</table>
<h2>Low-confidence or uncertain events</h2>
<p class="muted">Not counted as violations: below the alert threshold, or the view was blocked. Kept so they can be checked.</p>
<table><tr><th>Clip / time</th><th>Rule</th><th>Severity</th><th>Person</th><th>Conf.</th><th>Why</th><th>Evidence</th></tr>
{''.join(row(f) for f in review_sorted) or '<tr><td colspan=7>None.</td></tr>'}
</table>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out_path
