"""Scripted scenarios for the reasoning layer: no video or model needed.

Run:  python -m pytest tests -q     (or)     python tests/test_engine.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchen_hygiene.config import Config
from kitchen_hygiene.engine import HygieneEngine
from kitchen_hygiene.observations import PersonObs, ObjObs, FrameObs
from kitchen_hygiene.bayes import p_soap

W = H = 1000
ZONES = {
    "sink": [[0.0, 0.3], [0.25, 0.3], [0.25, 0.7], [0.0, 0.7]],
    "soap": [[0.25, 0.3], [0.3, 0.3], [0.3, 0.4], [0.25, 0.4]],
    "prep": [[0.35, 0.3], [0.6, 0.3], [0.6, 0.7], [0.35, 0.7]],
    "raw": [[0.62, 0.3], [0.72, 0.3], [0.72, 0.5], [0.62, 0.5]],
    "cooked": [[0.75, 0.3], [0.85, 0.3], [0.85, 0.5], [0.75, 0.5]],
    "used_rack": [[0.9, 0.0], [1.0, 0.0], [1.0, 0.2], [0.9, 0.2]],
    "clean_rack": [[0.9, 0.22], [1.0, 0.22], [1.0, 0.4], [0.9, 0.4]],
    "cooking": [[0.65, 0.75], [0.9, 0.75], [0.9, 1.0], [0.65, 1.0]],
    "storage": [[0.35, 0.0], [0.5, 0.0], [0.5, 0.15], [0.35, 0.15]],
}
FPS = 5.0


def cfg():
    c = Config()
    c.zones = ZONES
    c.sample_fps = FPS
    return c


def person(tid, hx, hy, k=0, together=True, visible=True, lather=None):
    """A person whose hands are at (hx, hy) (normalised). Jitter k makes rubbing motion."""
    jx = 8 * math.sin(k * 1.7)
    jy = 8 * math.cos(k * 1.3)
    x, y = hx * W, hy * H
    gap = 30 if together else 160
    c = 0.9 if visible else 0.0
    wrists = [(x - gap / 2 + jx, y + jy, c), (x + gap / 2 - jx, y - jy, c)]
    box = (x - 120, y - 300, x + 120, y + 250)
    return PersonObs(track_id=tid, box=box, conf=0.9, wrists=wrists, lather=lather)


class Sim:
    def __init__(self, c=None, classes=()):
        self.cfg = c or cfg()
        self.eng = HygieneEngine(self.cfg, W, H, available_classes=classes)
        self.t = 0.0
        self.i = 0

    def run(self, seconds, persons_fn, objects_fn=lambda k: []):
        n = int(round(seconds * FPS))
        for k in range(n):
            fo = FrameObs(t=self.t, frame_idx=self.i, width=W, height=H, persons=persons_fn(k), objects=objects_fn(k))
            self.eng.update(fo)
            self.t += 1 / FPS
            self.i += 1
        return self

    def done(self):
        self.eng.finish()
        return self.eng.findings


def find(fs, vtype, status=None):
    return [f for f in fs if f.vtype == vtype and (status is None or f.status == status)]


# ----------------------------------------------------------------------------- scenarios
def test_bayes_matches_cheat_sheet():
    c = Config()
    c.p_soap_prior = 0.7
    c.p_lather_given_soap = (0.8, 0.1)
    assert abs(p_soap(c, lather=True) - 0.949) < 0.005
    assert abs(p_soap(c, lather=False) - 0.341) < 0.005


def test_wash_with_soap_then_prep_is_compliant():
    s = Sim()
    s.run(1, lambda k: [person(1, 0.27, 0.35, k)])                                     # touch soap dispenser
    s.run(22, lambda k: [person(1, 0.12, 0.5, k, lather=0.02 + 0.01 * min(k, 20))])     # rub 22 s, lather builds
    s.run(2, lambda k: [person(1, 0.3, 0.8, k)])                       # walk
    s.run(8, lambda k: [person(1, 0.48, 0.5, k)])                      # prep
    fs = s.done()
    assert find(fs, "HANDWASH_VIOLATION", "compliant"), fs
    assert not find(fs, "HANDWASH_VIOLATION", "violation")


def test_no_wash_is_violation():
    s = Sim()
    s.run(2, lambda k: [])                                   # empty kitchen, then the cook walks in
    s.run(2, lambda k: [person(2, 0.3 + 0.0 * k, 0.92, k)])          # enters at the bottom edge
    s.run(8, lambda k: [person(2, 0.48, 0.5, k)])
    fs = s.done()
    v = find(fs, "HANDWASH_VIOLATION", "violation")
    assert v and v[0].person == 2 and v[0].confidence >= 0.5


def test_rub_without_soap_is_violation():
    s = Sim()
    s.run(22, lambda k: [person(3, 0.12, 0.5, k, lather=0.02)])     # no dispenser, no lather, 22 s
    s.run(8, lambda k: [person(3, 0.48, 0.5, k)])
    fs = s.done()
    v = find(fs, "HANDWASH_VIOLATION", "violation")
    assert v and v[0].detail == "no_soap" and v[0].confidence > 0.7, fs


def test_phone_after_wash_recontaminates():
    s = Sim()
    s.run(1, lambda k: [person(4, 0.27, 0.35, k)])
    s.run(22, lambda k: [person(4, 0.12, 0.5, k, lather=0.02 + 0.01 * min(k, 20))])
    phone = lambda k: [ObjObs("cell phone", 0.8, (300, 820, 340, 860))]
    s.run(2, lambda k: [person(4, 0.32, 0.84, k)], phone)
    s.run(8, lambda k: [person(4, 0.48, 0.5, k)])
    fs = s.done()
    v = find(fs, "HANDWASH_VIOLATION", "violation")
    assert v and "phone" in v[0].reason


def test_two_people_right_person_blamed():
    s = Sim()
    s.run(2, lambda k: [])
    s.run(1, lambda k: [person(5, 0.27, 0.35, k), person(6, 0.3, 0.9, k)])
    s.run(22, lambda k: [person(5, 0.12, 0.5, k, lather=0.02 + 0.01 * min(k, 20)), person(6, 0.45, 0.5, k)])
    s.run(8, lambda k: [person(5, 0.5, 0.55, k), person(6, 0.45, 0.5, k)])
    fs = s.done()
    assert [f.person for f in find(fs, "HANDWASH_VIOLATION", "violation")] == [6]
    assert [f.person for f in find(fs, "HANDWASH_VIOLATION", "compliant")] == [5]


def test_occlusion_near_sink_is_uncertain_not_violation():
    s = Sim()
    s.run(2, lambda k: [person(7, 0.12, 0.5, k)])            # at sink briefly
    s.run(5, lambda k: [])                                   # blocked from view
    s.run(8, lambda k: [person(7, 0.48, 0.5, k)])
    fs = s.done()
    assert find(fs, "HANDWASH_VIOLATION", "uncertain")
    assert not find(fs, "HANDWASH_VIOLATION", "violation")


def test_relink_keeps_wash_history():
    s = Sim()
    s.run(1, lambda k: [person(8, 0.27, 0.35, k)])
    s.run(22, lambda k: [person(8, 0.12, 0.5, k, lather=0.02 + 0.01 * min(k, 20))])
    s.run(1.4, lambda k: [])                                 # tracker loses the person ...
    s.run(1, lambda k: [person(99, 0.2, 0.55, k)])   # ... and gives a new ID nearby
    s.run(8, lambda k: [person(99, 0.48, 0.5, k)])
    fs = s.done()
    assert find(fs, "HANDWASH_VIOLATION", "compliant"), [(f.status, f.reason) for f in fs]


def test_vegetables_straight_to_board():
    s = Sim()
    s.run(2, lambda k: [person(9, 0.42, 0.08, k)])   # takes from storage
    s.run(8, lambda k: [person(9, 0.48, 0.5, k)])
    fs = s.done()
    assert find(fs, "VEGETABLE_WASH_VIOLATION", "violation")


def test_vegetables_rinsed_first():
    s = Sim()
    veg_in_sink = lambda k: [ObjObs("carrot", 0.7, (100, 480, 140, 520))]
    s.run(2, lambda k: [person(10, 0.42, 0.08, k)])
    s.run(5, lambda k: [person(10, 0.12, 0.5, k, together=False)], veg_in_sink)  # rinsing
    s.run(8, lambda k: [person(10, 0.48, 0.5, k)])
    fs = s.done()
    assert find(fs, "VEGETABLE_WASH_VIOLATION", "compliant")
    assert not find(fs, "VEGETABLE_WASH_VIOLATION", "violation")


def test_used_utensil_to_cooking():
    s = Sim()
    s.run(2, lambda k: [person(11, 0.95, 0.1, k)])
    s.run(3, lambda k: [person(11, 0.78, 0.88, k)])
    fs = s.done()
    assert find(fs, "DIRTY_UTENSIL_VIOLATION", "violation")


def test_clean_utensil_is_compliant():
    s = Sim()
    s.run(2, lambda k: [person(12, 0.95, 0.3, k)])
    s.run(3, lambda k: [person(12, 0.78, 0.88, k)])
    fs = s.done()
    assert find(fs, "DIRTY_UTENSIL_VIOLATION", "compliant")
    assert not find(fs, "DIRTY_UTENSIL_VIOLATION", "violation")


def test_raw_then_cooked_without_wash():
    s = Sim()
    s.run(3, lambda k: [person(13, 0.67, 0.4, k)])
    s.run(3, lambda k: [person(13, 0.8, 0.4, k)])
    fs = s.done()
    v = find(fs, "CROSS_CONTAMINATION_RISK", "violation")
    assert v and v[0].person == 13


def test_raw_wash_cooked_is_compliant():
    s = Sim()
    s.run(3, lambda k: [person(14, 0.67, 0.4, k)])
    s.run(1, lambda k: [person(14, 0.27, 0.35, k)])
    s.run(21, lambda k: [person(14, 0.12, 0.5, k, lather=0.02 + 0.01 * min(k, 20))])
    s.run(3, lambda k: [person(14, 0.8, 0.4, k)])
    fs = s.done()
    assert find(fs, "CROSS_CONTAMINATION_RISK", "compliant")
    assert not find(fs, "CROSS_CONTAMINATION_RISK", "violation")


def test_pest_sightings_merge_and_blips_ignored():
    s = Sim()
    roach = lambda k: [ObjObs("cockroach", 0.6, (500 + 15 * k, 500, 520 + 15 * k, 515))]   # it scurries
    s.run(2, lambda k: [])
    s.run(1.6, lambda k: [], roach)          # 8 frames
    s.run(1.0, lambda k: [])                 # gap 1 s (< 3 s merge gap)
    s.run(1.0, lambda k: [], roach)
    s.run(5, lambda k: [])
    s.run(0.2, lambda k: [], lambda k: [ObjObs("fly", 0.4, (10, 10, 14, 14))])   # one-frame blip
    s.run(5, lambda k: [])
    fs = s.done()
    pests = find(fs, "PEST_DETECTED")
    assert len(pests) == 1 and pests[0].detail == "cockroach" and pests[0].evidence["sightings"] == 13
    assert pests[0].severity == "High"
    assert any(e.etype == "PEST_BLIP_IGNORED" for e in s.eng.events)


def test_already_there_at_start_is_uncertain():
    s = Sim()
    s.run(8, lambda k: [person(20, 0.48, 0.5, k)])           # chopping from the very first frame
    fs = s.done()
    assert find(fs, "HANDWASH_VIOLATION", "uncertain")
    assert not find(fs, "HANDWASH_VIOLATION", "violation")


def test_static_lookalike_is_not_a_pest():
    s = Sim()
    s.run(10, lambda k: [], lambda k: [ObjObs("cockroach", 0.7, (700, 600, 720, 615))])   # never moves
    fs = s.done()
    assert not find(fs, "PEST_DETECTED")
    assert any(e.etype == "PEST_STATIC_IGNORED" for e in s.eng.events)


def test_calibrated_lookalike_spot_is_ignored():
    s = Sim()
    s.eng.cfg.pest_ignore_boxes = [(0.30, 0.50, 0.45, 0.65)]          # a spot flagged in the empty kitchen
    W, H = s.eng.w, s.eng.h
    jitter = lambda k: [ObjObs("cockroach", 0.7, (0.36 * W + 6 * (k % 3), 0.56 * H, 0.38 * W + 6 * (k % 3), 0.58 * H))]
    s.run(6, lambda k: [], jitter)                                     # jittery fake inside the spot
    s.run(4, lambda k: [], lambda k: [ObjObs("cockroach", 0.6, (0.6 * W + 15 * k, 0.8 * H, 0.62 * W + 15 * k, 0.82 * H))])
    fs = s.done()
    assert len(find(fs, "PEST_DETECTED")) == 1                          # only the real, moving one outside the spot
    assert s.eng.pest_calibration_skips > 0


def test_calibration_clusters_repeated_spots_only():
    from kitchen_hygiene.calibrate import cluster_spots
    stove = (0.70, 0.70, 0.74, 0.74)
    per_image = [[stove], [(0.705, 0.70, 0.745, 0.745)], [stove, (0.20, 0.30, 0.22, 0.32)]]   # one-off box in image 3
    spots = cluster_spots(per_image, min_votes=2)
    assert len(spots) == 1 and spots[0][0] < 0.70 and spots[0][2] > 0.745


def test_spill_not_cleaned():
    s = Sim()
    spill = lambda k: [ObjObs("spill", 0.7, (450, 450, 520, 500))]
    s.run(35, lambda k: [], spill)
    fs = s.done()
    assert find(fs, "SURFACE_NOT_CLEANED", "violation")


def test_spill_cleaned_in_time():
    s = Sim()
    spill = lambda k: [ObjObs("spill", 0.7, (450, 450, 520, 500))]
    s.run(12, lambda k: [], spill)
    s.run(6, lambda k: [])
    fs = s.done()
    assert find(fs, "SURFACE_NOT_CLEANED", "compliant") and not find(fs, "SURFACE_NOT_CLEANED", "violation")


def test_ppe_bare_head():
    s = Sim(classes={"cap", "no_cap"})
    bare = lambda k: [ObjObs("no_cap", 0.7, (380, 210, 580, 300))]
    s.run(2, lambda k: [person(15, 0.3, 0.85, k)])
    s.run(8, lambda k: [person(15, 0.48, 0.5, k)], bare)
    fs = s.done()
    assert find(fs, "PPE_VIOLATION", "violation")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
