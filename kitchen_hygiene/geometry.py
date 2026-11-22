"""Zones, point-in-polygon tests and drawing helpers."""
import cv2
import numpy as np

ZONE_COLOURS = {
    "sink": (230, 160, 40), "soap": (255, 220, 120), "prep": (60, 180, 75), "raw": (60, 60, 220),
    "cooked": (40, 170, 240), "used_rack": (40, 40, 160), "clean_rack": (150, 200, 120),
    "cooking": (0, 120, 255), "storage": (180, 120, 200), "used_oil": (20, 90, 160),
}


class Zones:
    def __init__(self, zones_norm: dict, width: int, height: int):
        self.norm = zones_norm
        self.w, self.h = width, height
        self.px = {name: np.array([[x * width, y * height] for x, y in pts], dtype=np.float32)
                   for name, pts in zones_norm.items()}

    def contains(self, name, point) -> bool:
        poly = self.px.get(name)
        if poly is None or point is None:
            return False
        return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0

    def which(self, point):
        return [n for n in self.px if self.contains(n, point)]

    def box_in(self, name, box) -> bool:
        x1, y1, x2, y2 = box
        return self.contains(name, ((x1 + x2) / 2, (y1 + y2) / 2))

    def has(self, name):
        return name in self.px


def draw_grid(img, step=0.1):
    """Overlay a labelled 10% grid so zone coordinates can be read off a frame."""
    out = img.copy()
    h, w = out.shape[:2]
    for i in range(1, int(1 / step)):
        x, y = int(i * step * w), int(i * step * h)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(out, f"{i * step:.1f}", (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(out, f"{i * step:.1f}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def draw_zones(img, zones: Zones, alpha=0.25):
    out = img.copy()
    overlay = img.copy()
    for name, poly in zones.px.items():
        c = ZONE_COLOURS.get(name, (200, 200, 200))
        pts = poly.astype(np.int32)
        cv2.fillPoly(overlay, [pts], c)
        cv2.polylines(out, [pts], True, c, 2)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(out, name, (int(x) + 4, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def draw_obs(img, fobs, zones=None, highlight_id=None, banner=None):
    out = draw_zones(img, zones) if zones is not None else img.copy()
    for o in fobs.objects:
        x1, y1, x2, y2 = map(int, o.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(out, f"{o.cls} {o.conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
    for p in fobs.persons:
        x1, y1, x2, y2 = map(int, p.box)
        col = (0, 0, 255) if p.track_id == highlight_id else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if p.track_id == highlight_id else 2)
        ty = y1 - 6 if y1 > 60 else y1 + 24
        cv2.putText(out, f"P{p.track_id:02d}", (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        for x, y, c in p.wrists:
            if c > 0.3:
                cv2.circle(out, (int(x), int(y)), 6, (255, 0, 255), -1)
    if banner:
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(out, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 120, 255), 2)
    return out


def iou(a, b):
    """Intersection over union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlap_frac(inner, outer):
    """Share of `inner`'s area that lies inside `outer`."""
    ix1, iy1, ix2, iy2 = max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area
