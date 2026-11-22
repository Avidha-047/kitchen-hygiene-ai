"""All tunable settings in one place.

Zones are polygons in *normalised* image coordinates (0..1), so they survive a change of
video resolution. Mark them once per camera position (see notebook step 3).
"""
from dataclasses import dataclass, field
import json


# Severity levels (deliverable 11 / Task 11 of the brief)
SEVERITY = {
    "HANDWASH_VIOLATION": "High",
    "CROSS_CONTAMINATION_RISK": "High",
    "DIRTY_UTENSIL_VIOLATION": "High",
    "VEGETABLE_WASH_VIOLATION": "Medium",
    "SURFACE_NOT_CLEANED": "Medium",
    "OIL_REUSE_FLAG": "Medium",
    "PPE_VIOLATION": "Low",
    "PEST_DETECTED:fly": "Medium",
    "PEST_DETECTED:cockroach": "High",
    "PEST_DETECTED:rodent": "Critical",
}
SEVERITY_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


@dataclass
class Config:
    # ---- models -------------------------------------------------------------
    pose_model: str = "yolov8n-pose.pt"          # people + wrist keypoints, tracked with ByteTrack
    object_model: str = "yolov8n.pt"             # COCO objects: sink, knife, bowl, carrot, broccoli, cell phone ...
    custom_models: list = field(default_factory=list)  # fine-tuned YOLO .pt files (pests, PPE, spill, ...)
    tracker: str = "bytetrack.yaml"

    # ---- sampling -----------------------------------------------------------
    sample_fps: float = 5.0                      # frames analysed per second of video
    imgsz: int = 640
    person_conf: float = 0.35
    object_conf: float = 0.30
    keypoint_conf: float = 0.30

    # ---- zones (normalised polygons) ----------------------------------------
    # Required: sink, prep. Optional: soap, raw, cooked, used_rack, clean_rack, cooking, storage, used_oil
    zones: dict = field(default_factory=dict)

    # ---- action recognition (rule-based temporal layer) ---------------------
    rub_max_wrist_gap: float = 0.60              # wrists closer than this x person-box width = hands together
    rub_min_motion: float = 0.015                # mean wrist movement per sample, as a fraction of box height
    wash_min_s: float = 20.0                     # WHO: rub for at least 20 s
    wash_min_detect_s: float = 3.0               # shorter rubbing is ignored as noise
    prep_min_s: float = 2.0                      # hands in prep zone this long = food prep started
    zone_min_s: float = 0.6                      # minimum dwell to count as a zone visit (removes flicker)
    session_gap_s: float = 10.0                  # leave prep zone this long = new prep session
    contact_px_frac: float = 0.08                # wrist-to-object distance counted as "touching" (fraction of width)

    # ---- temporal rules ------------------------------------------------------
    cross_contam_window_s: float = 300.0
    sequence_window_s: float = 60.0              # used rack -> cooking, used oil -> cooking, storage -> prep
    surface_limit_s: float = 30.0                # spill must be wiped within this
    pest_merge_gap_s: float = 3.0                # sightings closer than this merge into one event
    pest_min_frames: int = 2                     # a single-frame blip is not a pest
    pest_min_move: float = 0.03                  # a pest must move at least this fraction of the frame width; static look-alikes are ignored
    pest_link_dist: float = 0.08                 # detections closer than this (fraction of width) belong to the same moving pest
    history_grace_s: float = 1.0                 # someone already in view this soon after the video starts has an unknown history
    entry_edge_frac: float = 0.15                # people walking in are first seen within this fraction of the left, right or bottom edge
    occlusion_gap_s: float = 2.0                 # track unseen this long near the sink = uncertain
    relink_max_gap_s: float = 3.0                # re-attach a new track ID to a just-lost one
    relink_max_dist: float = 0.20                # ... if its centre is this close (fraction of width)

    # ---- Bayesian network for "was soap used?" ------------------------------
    p_soap_prior: float = 0.70
    p_dispenser_given_soap: tuple = (0.85, 0.10)   # P(touched dispenser | soap), P(... | no soap)
    p_lather_given_soap: tuple = (0.80, 0.10)
    p_longrub_given_soap: tuple = (0.75, 0.35)
    soap_threshold: float = 0.50                 # below this P(soap) the wash does not count
    lather_white_frac: float = 0.06              # fraction of near-white pixels around the hands = lather seen

    # ---- class names -----------------------------------------------------------
    pest_classes: dict = field(default_factory=lambda: {
        "cockroach": "cockroach", "rat": "rodent", "rodent": "rodent", "mouse_animal": "rodent", "fly": "fly"})
    vegetable_classes: tuple = ("carrot", "broccoli", "vegetable", "potato", "tomato", "onion")
    phone_classes: tuple = ("cell phone",)
    head_cover_classes: tuple = ("cap", "hairnet", "head_cover")
    no_head_cover_classes: tuple = ("no_cap", "no_hairnet", "bare_head")
    spill_classes: tuple = ("spill",)
    ppe_min_frac: float = 0.6                    # head cover must be seen in >= 60% of prep frames

    # ---- alerting ---------------------------------------------------------------
    thresholds: dict = field(default_factory=lambda: {})   # per-violation confidence thresholds (tuned on validation)
    default_threshold: float = 0.50
    clip_pad_s: float = 2.0

    def threshold(self, vtype: str) -> float:
        return self.thresholds.get(vtype, self.thresholds.get(vtype.split(":")[0], self.default_threshold))

    # ---- persistence -------------------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
        return cfg


def severity_of(vtype: str, detail: str = "") -> str:
    key = f"{vtype}:{detail}" if detail and f"{vtype}:{detail}" in SEVERITY else vtype
    return SEVERITY.get(key, "Medium")
