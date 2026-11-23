"""Reasoning layer: temporal event engine + hygiene rule engine.

Input : one FrameObs per sampled frame (people with IDs + wrists, objects).
Output: Events (what happened, when, to whom) and Findings (rule verdicts with evidence).

The agent is model-based (course Unit 1): it keeps a state per person (where their hands
have been, when they last washed, whether they touched raw food or a phone since), because a
single frame cannot show whether a violation happened.

Every rule is a first-order statement over people p and times t (Units 4-6), e.g.
  forall p: InPrepZone(p) ^ Prepares(p, t2) ^ not exists t1 (Washes(p, t1) ^ t1 < t2) => Violation(p)
and is applied by forward chaining (modus ponens) whenever its trigger event occurs.
"""
from dataclasses import dataclass, field
from typing import Optional
import math

from .calibrate import in_ignored_spot
from .geometry import Zones, overlap_frac
from .bayes import p_soap, explain
from .config import severity_of


# ----------------------------------------------------------------------------- data
@dataclass
class Event:
    etype: str
    person: Optional[int]
    t_start: float
    t_end: float
    info: dict = field(default_factory=dict)


@dataclass
class Finding:
    vtype: str                  # e.g. HANDWASH_VIOLATION
    status: str                 # violation | compliant | uncertain
    person: Optional[int]
    t_start: float
    t_end: float
    confidence: float
    reason: str
    evidence: dict = field(default_factory=dict)
    detail: str = ""            # sub-type, e.g. pest kind
    frame_idx: Optional[int] = None

    @property
    def severity(self):
        return severity_of(self.vtype, self.detail)


@dataclass
class Visit:
    zone: str
    t_in: float
    t_last: float
    frames: int = 1
    veg_seen: bool = False
    t_out: Optional[float] = None


@dataclass
class Wash:
    t_start: float
    t_end: float
    p_soap: float
    dispenser: Optional[bool]
    lather: Optional[bool]
    longrub: bool
    hidden_frac: float
    explanation: str

    @property
    def duration(self):
        return self.t_end - self.t_start


class PersonState:
    def __init__(self, pid, t):
        self.pid = pid
        self.first_t = t
        self.last_t = t
        self.last_centre = None
        self.first_centre = None
        self.seen_times = []
        self.open_visits = {}
        self.visits = []
        self.gaps = []                  # (t_from, t_to, near_sink)
        self.last_zones = set()
        # rubbing run
        self.rub = None
        self.washes = []
        self.last_wrists = None
        # food prep
        self.prep_since = None
        self.prep_last = None
        self.session = None             # dict(start, last, cover, bare, frames)
        self.phone_touches = []

    def zone_visits(self, zone, before=None, after=None):
        out = [v for v in self.visits + list(self.open_visits.values()) if v.zone == zone]
        if before is not None:
            out = [v for v in out if v.t_in <= before]
        if after is not None:
            out = [v for v in out if (v.t_out or v.t_last) >= after]
        return sorted(out, key=lambda v: v.t_in)


# ----------------------------------------------------------------------------- engine
class HygieneEngine:
    def __init__(self, cfg, width, height, available_classes=()):
        self.cfg = cfg
        self.zones = Zones(cfg.zones, width, height)
        self.w, self.h = width, height
        self.pest_calibration_skips = 0
        self.dt = 1.0 / cfg.sample_fps
        self.people = {}
        self.alias = {}                 # tracker id -> person id (re-linking after a lost track)
        self.events = []
        self.findings = []
        self.pests = {}                 # kind -> open sighting dict
        self.spill = None
        names = set(available_classes)
        self.ppe_enabled = bool(names & (set(cfg.head_cover_classes) | set(cfg.no_head_cover_classes)))
        self.cover_positive = bool(names & set(cfg.head_cover_classes))
        self.t = 0.0
        self.t0 = None                  # time of the first frame: people present then have an unseen history
        self.frame_idx = 0

    # ------------------------------------------------------------------ helpers
    def _emit(self, finding):
        finding.frame_idx = self.frame_idx if finding.frame_idx is None else finding.frame_idx
        self.findings.append(finding)
        self._new.append(finding)

    def _visibility(self, ps, t_from, t_to):
        t_from = max(t_from, ps.first_t)
        if t_to - t_from < self.dt:
            return 1.0
        n = sum(1 for s in ps.seen_times if t_from <= s <= t_to)
        return min(1.0, n / ((t_to - t_from) / self.dt + 1))

    def _resolve_id(self, tid, centre, t):
        if tid in self.alias:
            return self.alias[tid]
        if tid in self.people:
            return tid
        # new tracker ID: re-link to a person lost moments ago near the same place
        best, best_d = None, None
        for pid, ps in self.people.items():
            lost_for = t - ps.last_t
            if self.dt * 1.5 < lost_for <= self.cfg.relink_max_gap_s and ps.last_centre is not None:
                d = math.dist(centre, ps.last_centre) / self.w
                if d <= self.cfg.relink_max_dist and (best_d is None or d < best_d):
                    best, best_d = pid, d
        if best is not None:
            self.alias[tid] = best
            self.events.append(Event("TRACK_RELINKED", best, t, t, {"new_tracker_id": tid}))
            return best
        self.people[tid] = PersonState(tid, t)
        return tid

    # ------------------------------------------------------------------ main step
    def update(self, fobs):
        self._new = []
        self.t, self.frame_idx = fobs.t, fobs.frame_idx
        if self.t0 is None:
            self.t0 = fobs.t
        seen = set()
        for p in fobs.persons:
            pid = self._resolve_id(p.track_id, p.centre, fobs.t)
            if pid in seen:
                continue
            seen.add(pid)
            self._update_person(self.people[pid], p, fobs)
        for pid, ps in self.people.items():
            if pid not in seen:
                self._close_stale(ps, fobs.t)
        self._update_pests(fobs)
        self._update_spill(fobs)
        return self._new

    def _update_person(self, ps, p, fobs):
        cfg, t = self.cfg, fobs.t
        if t - ps.last_t > cfg.occlusion_gap_s:
            near_sink = "sink" in ps.last_zones
            ps.gaps.append((ps.last_t, t, near_sink))
        ps.last_t = t
        ps.seen_times.append(t)
        if ps.first_centre is None:
            ps.first_centre = p.centre
        ps.last_centre = p.centre

        wrists = p.visible_wrists(cfg.keypoint_conf)
        hands_hidden = len(wrists) == 0
        points = wrists if wrists else [p.centre]
        zones_now = set()
        for pt in points:
            zones_now.update(self.zones.which(pt))
        if hands_hidden and "sink" in ps.last_zones and self.zones.contains("sink", p.centre):
            zones_now.add("sink")
        near_sink_body = self.zones.contains("sink", p.centre)

        # actions first, so a wash that just ended is known before the next zone's rules fire
        self._update_rubbing(ps, p, wrists, hands_hidden, fobs)

        # zone visits (with hysteresis so one missed frame does not split a visit)
        for z in zones_now:
            v = ps.open_visits.get(z)
            if v is None:
                v = ps.open_visits[z] = Visit(z, t, t)
                self._on_visit_open(ps, v)
            else:
                v.t_last, v.frames = t, v.frames + 1
            if z == "sink" and any(o.cls in cfg.vegetable_classes and self.zones.box_in("sink", o.box) for o in fobs.objects):
                v.veg_seen = True
        for z in list(ps.open_visits):
            if z not in zones_now and t - ps.open_visits[z].t_last > cfg.zone_min_s:
                self._close_visit(ps, z)
        ps.last_zones = zones_now | ({"sink"} if hands_hidden and near_sink_body else set())

        # phone touch = re-contamination after washing
        for o in fobs.objects:
            if o.cls in cfg.phone_classes and wrists:
                if min(math.dist(o.centre, w) for w in wrists) < cfg.contact_px_frac * self.w:
                    if not ps.phone_touches or t - ps.phone_touches[-1] > 2.0:
                        ps.phone_touches.append(t)
                        self.events.append(Event("PHONE_TOUCH", ps.pid, t, t))

        self._update_prep(ps, p, zones_now, fobs)

    def _close_stale(self, ps, t):
        for z in list(ps.open_visits):
            if t - ps.open_visits[z].t_last > self.cfg.zone_min_s:
                self._close_visit(ps, z)
        if ps.rub and t - ps.rub["last"] > 1.0:
            self._finish_rub(ps)
        self._maybe_end_session(ps, t)

    def _close_visit(self, ps, z):
        v = ps.open_visits.pop(z)
        v.t_out = v.t_last
        if v.frames >= 2:
            ps.visits.append(v)
            self.events.append(Event(f"VISIT_{z.upper()}", ps.pid, v.t_in, v.t_out, {"veg_seen": v.veg_seen}))

    # ------------------------------------------------------------------ actions
    def _update_rubbing(self, ps, p, wrists, hands_hidden, fobs):
        cfg, t = self.cfg, fobs.t
        at_sink = self.zones.contains("sink", p.centre) or any(self.zones.contains("sink", w) for w in wrists)
        together = False
        motion = 0.0
        if len(wrists) == 2:
            gap = math.dist(wrists[0], wrists[1]) / p.width
            together = gap <= cfg.rub_max_wrist_gap and all(self.zones.contains("sink", w) for w in wrists)
            if ps.last_wrists and len(ps.last_wrists) == 2:
                motion = (math.dist(wrists[0], ps.last_wrists[0]) + math.dist(wrists[1], ps.last_wrists[1])) / (2 * p.height)
        ps.last_wrists = wrists if len(wrists) == 2 else None
        rubbing = together or (hands_hidden and at_sink and ps.rub is not None)
        if rubbing:
            if ps.rub is None:
                ps.rub = dict(start=t, last=t, n=0, hidden=0, motion=[], lather=[], disp=False)
            r = ps.rub
            r["last"], r["n"] = t, r["n"] + 1
            r["hidden"] += 1 if hands_hidden else 0
            if not hands_hidden:
                r["motion"].append(motion)
            if p.lather is not None:
                r["lather"].append(p.lather)
        elif ps.rub and (t - ps.rub["last"] > 1.0 or (wrists and not at_sink)):
            self._finish_rub(ps)
        if ps.rub and self.zones.has("soap") and any(self.zones.contains("soap", w) for w in wrists):
            ps.rub["disp"] = True
        if self.zones.has("soap") and any(self.zones.contains("soap", w) for w in wrists):
            ps.last_dispenser_t = t

    def _finish_rub(self, ps):
        cfg, r = self.cfg, ps.rub
        ps.rub = None
        duration = r["last"] - r["start"]
        motion = sum(r["motion"]) / len(r["motion"]) if r["motion"] else 0.0
        if duration < cfg.wash_min_detect_s or (r["motion"] and motion < cfg.rub_min_motion):
            return
        dispenser = None
        if self.zones.has("soap"):
            recent = getattr(ps, "last_dispenser_t", None)
            dispenser = r["disp"] or (recent is not None and r["start"] - 8.0 <= recent <= r["last"])
        lather = None
        if len(r["lather"]) >= 4:
            base = min(r["lather"][:3])
            peak = sorted(r["lather"])[int(0.75 * (len(r["lather"]) - 1))]
            lather = (peak - base) >= cfg.lather_white_frac or peak >= 3 * cfg.lather_white_frac
        longrub = duration >= cfg.wash_min_s
        ps_ = p_soap(cfg, dispenser, lather, longrub)
        w = Wash(r["start"], r["last"], ps_, dispenser, lather, longrub, r["hidden"] / max(1, r["n"]),
                 explain(cfg, dispenser, lather, longrub))
        ps.washes.append(w)
        self.events.append(Event("HANDWASH", ps.pid, w.t_start, w.t_end,
                                 {"p_soap": round(ps_, 3), "explanation": w.explanation, "duration_s": round(duration, 1)}))

    def _update_prep(self, ps, p, zones_now, fobs):
        cfg, t = self.cfg, fobs.t
        if "prep" in zones_now:
            if ps.prep_since is None or (ps.prep_last is not None and t - ps.prep_last > 1.0):
                ps.prep_since = t
            ps.prep_last = t
            if ps.session is None and t - ps.prep_since >= cfg.prep_min_s:
                ps.session = dict(start=ps.prep_since, last=t, cover=0, bare=0, frames=0, frame_idx=fobs.frame_idx)
                self.events.append(Event("FOOD_PREP_START", ps.pid, ps.prep_since, ps.prep_since))
                self._rule_handwash(ps, ps.prep_since)
                self._rule_vegetable(ps, ps.prep_since)
        if ps.session is not None:
            if "prep" in zones_now:
                ps.session["last"] = t
                if self.ppe_enabled:
                    ps.session["frames"] += 1
                    for o in fobs.objects:
                        if overlap_frac(o.box, p.head_box) > 0.3 or overlap_frac(p.head_box, o.box) > 0.3:
                            if o.cls in cfg.head_cover_classes:
                                ps.session["cover"] += 1
                                break
                            if o.cls in cfg.no_head_cover_classes:
                                ps.session["bare"] += 1
                                break
            self._maybe_end_session(ps, t)

    def _maybe_end_session(self, ps, t, force=False):
        s = ps.session
        if s is None:
            return
        if force or t - s["last"] > self.cfg.session_gap_s:
            ps.session = None
            self.events.append(Event("FOOD_PREP_END", ps.pid, s["start"], s["last"]))
            self._rule_ppe(ps, s)

    # ------------------------------------------------------------------ rules
    def _rule_handwash(self, ps, t2):
        cfg = self.cfg
        washes = [w for w in ps.washes if w.t_end <= t2 + 1.0]
        if ps.rub and t2 - ps.rub["start"] >= cfg.wash_min_detect_s:   # still rubbing when prep began
            self._finish_rub(ps)
            washes = [w for w in ps.washes if w.t_end <= t2 + 1.0]
        last = washes[-1] if washes else None
        recontam = []
        if last:
            recontam += [f"raw food at {v.t_in:.1f}s" for v in ps.zone_visits("raw") if last.t_end <= v.t_in <= t2]
            recontam += [f"phone at {x:.1f}s" for x in ps.phone_touches if last.t_end <= x <= t2]
        window_start = max(ps.first_t, t2 - 60.0)
        vis = self._visibility(ps, window_start, t2)
        occluded = [g for g in ps.gaps if g[2] and g[1] >= t2 - 120.0]
        hidden_wash = last is not None and last.hidden_frac > 0.5
        ev = {"food_prep_start": round(t2, 1), "visibility": round(vis, 2),
              "washes_before": [dict(start=round(w.t_start, 1), end=round(w.t_end, 1), p_soap=round(w.p_soap, 2)) for w in washes]}
        if last is None or recontam:
            reason = "Moved to food preparation with no hand-wash event detected" if last is None else \
                     f"Washed at {last.t_start:.1f}s but then touched {', '.join(recontam)} before food prep"
            # never seen arriving: in view from the first frames, or first seen away from the frame edges (appeared mid-room)
            fx, fy = (ps.first_centre or (0, 0))
            m = cfg.entry_edge_frac
            entered_from_edge = fx < m * self.w or fx > (1 - m) * self.w or fy > (1 - m) * self.h
            already_there = (ps.first_t - (self.t0 or 0.0) <= cfg.history_grace_s or not entered_from_edge) \
                and not ps.zone_visits("sink", before=t2)
            if last is None and already_there:
                self._emit(Finding("HANDWASH_VIOLATION", "uncertain", ps.pid, ps.first_t, t2, 0.3,
                                   "Never seen arriving (already in the room when first seen), so we cannot tell whether they washed earlier", ev))
                return
            if last is None and (occluded or ps.first_t > t2 - 3.0 and vis < 0.6):
                self._emit(Finding("HANDWASH_VIOLATION", "uncertain", ps.pid, window_start, t2, 0.35 * vis,
                                   reason + "; the person was out of view near the sink, so a wash may have been missed", ev))
                return
            conf = 0.55 + 0.4 * vis
            self._emit(Finding("HANDWASH_VIOLATION", "violation", ps.pid, window_start, t2, round(conf, 3), reason, ev))
            return
        ev.update(last_wash=dict(start=round(last.t_start, 1), end=round(last.t_end, 1)), bayes=last.explanation)
        if last.p_soap < cfg.soap_threshold:
            status = "uncertain" if hidden_wash else "violation"
            self._emit(Finding("HANDWASH_VIOLATION", status, ps.pid, last.t_start, t2, round(1 - last.p_soap, 3),
                               f"Hands rubbed at the sink but soap use is unlikely ({last.explanation})", ev, detail="no_soap"))
        else:
            self._emit(Finding("HANDWASH_VIOLATION", "compliant", ps.pid, last.t_start, t2, round(last.p_soap, 3),
                               f"Washed with soap before food prep ({last.explanation})", ev))

    def _rule_vegetable(self, ps, t2):
        if not self.zones.has("storage"):
            return
        cfg = self.cfg
        took = [v for v in ps.zone_visits("storage", before=t2) if t2 - (v.t_out or v.t_last) <= cfg.sequence_window_s]
        if not took:
            return
        v_store = took[-1]
        t1 = v_store.t_out or v_store.t_last
        sinks = [v for v in ps.zone_visits("sink") if v.t_in >= t1 - 0.5 and v.t_in <= t2]
        washed_veg = [v for v in sinks if v.veg_seen]
        rinsed = [v for v in sinks if not any(w.t_start >= v.t_in - 0.5 and w.t_end <= (v.t_out or v.t_last) + 0.5 for w in ps.washes)]
        ev = {"took_from_storage": round(v_store.t_in, 1), "food_prep_start": round(t2, 1),
              "sink_visits_between": [round(v.t_in, 1) for v in sinks]}
        if washed_veg or rinsed:
            conf = 0.8 if washed_veg else 0.6
            self._emit(Finding("VEGETABLE_WASH_VIOLATION", "compliant", ps.pid, v_store.t_in, t2, conf,
                               "Vegetables went to the sink before the board", ev))
        else:
            vis = self._visibility(ps, v_store.t_in, t2)
            self._emit(Finding("VEGETABLE_WASH_VIOLATION", "violation", ps.pid, v_store.t_in, t2, round(0.5 + 0.4 * vis, 3),
                               "Took vegetables from storage and started preparing them with no visit to the sink", ev))

    def _on_visit_open(self, ps, v):
        cfg, t = self.cfg, v.t_in
        if v.zone == "cooking":
            for src, vtype, label in (("used_rack", "DIRTY_UTENSIL_VIOLATION", "a utensil from the used rack"),
                                      ("used_oil", "OIL_REUSE_FLAG", "oil from the USED container")):
                if not self.zones.has(src):
                    continue
                prev = [x for x in ps.zone_visits(src, before=t) if t - (x.t_out or x.t_last) <= cfg.sequence_window_s and x is not v]
                if not prev:
                    continue
                t1 = prev[-1].t_out or prev[-1].t_last
                washed = [s for s in ps.zone_visits("sink") if t1 - 0.5 <= s.t_in <= t]
                ev = {"from_zone": src, "picked_at": round(prev[-1].t_in, 1), "reached_cooking": round(t, 1)}
                vis = self._visibility(ps, prev[-1].t_in, t)
                if src == "used_rack" and washed:
                    self._emit(Finding(vtype, "compliant", ps.pid, prev[-1].t_in, t, 0.7, "Used utensil went to the sink first", ev))
                else:
                    self._emit(Finding(vtype, "violation", ps.pid, prev[-1].t_in, t, round(0.5 + 0.4 * vis, 3),
                                       f"Took {label} straight to the cooking area", ev))
            if self.zones.has("clean_rack"):
                prev = [x for x in ps.zone_visits("clean_rack", before=t) if t - (x.t_out or x.t_last) <= cfg.sequence_window_s]
                if prev and not [x for x in ps.zone_visits("used_rack", before=t) if t - (x.t_out or x.t_last) <= cfg.sequence_window_s]:
                    self._emit(Finding("DIRTY_UTENSIL_VIOLATION", "compliant", ps.pid, prev[-1].t_in, t, 0.7,
                                       "Utensil came from the clean rack", {"picked_at": round(prev[-1].t_in, 1)}))
        if v.zone == "cooked" and self.zones.has("raw"):
            raws = [x for x in ps.zone_visits("raw", before=t) if t - (x.t_out or x.t_last) <= cfg.cross_contam_window_s and x is not v]
            if raws:
                t1 = raws[-1].t_out or raws[-1].t_last
                cleaned = [w for w in ps.washes if w.t_start >= t1 - 0.5 and w.t_end <= t + 0.5]
                ev = {"raw_handled": round(raws[-1].t_in, 1), "cooked_handled": round(t, 1),
                      "gap_s": round(t - t1, 1), "washes_between": len(cleaned)}
                if cleaned:
                    self._emit(Finding("CROSS_CONTAMINATION_RISK", "compliant", ps.pid, raws[-1].t_in, t, 0.75,
                                       "Hands washed between raw and cooked food", ev))
                else:
                    vis = self._visibility(ps, raws[-1].t_in, t)
                    self._emit(Finding("CROSS_CONTAMINATION_RISK", "violation", ps.pid, raws[-1].t_in, t, round(0.5 + 0.45 * vis, 3),
                                       f"Handled raw food, then cooked food {t - t1:.0f}s later with no hand-wash in between", ev))

    def _rule_ppe(self, ps, s):
        if not self.ppe_enabled or s["frames"] < 3:
            return
        cover_frac = s["cover"] / s["frames"]
        bare_frac = s["bare"] / s["frames"]
        ev = {"prep_frames": s["frames"], "head_cover_seen_frac": round(cover_frac, 2), "bare_head_seen_frac": round(bare_frac, 2)}
        violated = bare_frac >= 0.5 or (self.cover_positive and cover_frac < self.cfg.ppe_min_frac and bare_frac > 0)
        if not violated and self.cover_positive and cover_frac < 0.2:
            violated = True
        if violated:
            conf = max(bare_frac, 1 - cover_frac) if self.cover_positive else bare_frac
            self._emit(Finding("PPE_VIOLATION", "violation", ps.pid, s["start"], s["last"], round(conf, 3),
                               "Prepared food without a head cover", ev, frame_idx=s.get("frame_idx")))
        else:
            self._emit(Finding("PPE_VIOLATION", "compliant", ps.pid, s["start"], s["last"], round(cover_frac, 3),
                               "Head cover worn during food prep", ev, frame_idx=s.get("frame_idx")))

    # ------------------------------------------------------------------ pests & spills (scene-level)
    def _update_pests(self, fobs):
        """Each pest detection joins the nearest open tracklet of its kind; a tracklet that never moves is a look-alike."""
        cfg, t = self.cfg, fobs.t
        for o in fobs.objects:
            kind = cfg.pest_classes.get(o.cls)
            if not kind:
                continue
            if cfg.pest_ignore_boxes and in_ignored_spot(o.centre, (self.w, self.h), cfg.pest_ignore_boxes):
                self.pest_calibration_skips += 1          # a look-alike spot found in the empty kitchen
                continue
            c = (o.centre[0] / self.w, o.centre[1] / self.w)
            best, best_d = None, None
            for k, s in self.pests.items():
                if s["kind"] != kind or t - s["last"] > cfg.pest_merge_gap_s or s["last"] == t:
                    continue
                d = math.dist(c, s["path"][-1])
                if d <= cfg.pest_link_dist and (best_d is None or d < best_d):
                    best, best_d = k, d
            if best is None:
                best = f"{kind}-{len(self.events)}-{t}-{len(self.pests)}"
                self.pests[best] = dict(kind=kind, first=t, last=t, n=0, confs=[], best_conf=0.0, path=[],
                                        frame_idx=fobs.frame_idx, zones=self.zones.which(o.centre))
            s = self.pests[best]
            s["last"], s["n"] = t, s["n"] + 1
            s["confs"].append(o.conf)
            s["path"].append(c)
            if o.conf > s["best_conf"]:
                s["best_conf"], s["frame_idx"] = o.conf, fobs.frame_idx
        for k in list(self.pests):
            if t - self.pests[k]["last"] > cfg.pest_merge_gap_s:
                self._close_pest(k)

    def _close_pest(self, key):
        s = self.pests.pop(key)
        kind = s["kind"]
        if s["n"] < self.cfg.pest_min_frames:
            self.events.append(Event("PEST_BLIP_IGNORED", None, s["first"], s["last"], {"kind": kind, "conf": round(s["best_conf"], 2)}))
            return
        moved = max(math.dist(s["path"][0], q) for q in s["path"])
        if moved < self.cfg.pest_min_move:
            self.events.append(Event("PEST_STATIC_IGNORED", None, s["first"], s["last"],
                                     {"kind": kind, "conf": round(s["best_conf"], 2), "moved": round(moved, 3)}))
            return
        # one pest seen as several tracklets at overlapping times is still one event
        for f in self.findings:
            if f.vtype == "PEST_DETECTED" and f.detail == kind and f.t_start - self.cfg.pest_merge_gap_s <= s["last"] and s["first"] <= f.t_end + self.cfg.pest_merge_gap_s:
                f.t_start, f.t_end = min(f.t_start, s["first"]), max(f.t_end, s["last"])
                f.evidence["sightings"] += s["n"]
                f.evidence["duration_s"] = round(f.t_end - f.t_start, 1)
                return
        conf = sum(s["confs"]) / len(s["confs"])
        conf = min(1.0, conf + 0.05 * min(4, s["n"] - 1))   # persistence across frames raises confidence
        self._emit(Finding("PEST_DETECTED", "violation", None, s["first"], s["last"], round(conf, 3),
                           f"{kind.capitalize()} seen moving in {s['n']} frames over {s['last'] - s['first']:.1f}s",
                           {"kind": kind, "sightings": s["n"], "duration_s": round(s["last"] - s["first"], 1),
                            "moved_frac_width": round(moved, 3), "zones": s["zones"]},
                           detail=kind, frame_idx=s["frame_idx"]))

    def _update_spill(self, fobs):
        cfg, t = self.cfg, fobs.t
        spills = [o for o in fobs.objects if o.cls in cfg.spill_classes and (not self.zones.has("prep") or self.zones.box_in("prep", o.box))]
        s = self.spill
        if spills:
            if s is None or t - s["last"] > 3.0:
                if s is not None:
                    self._close_spill()
                self.spill = s = dict(first=t, last=t, flagged=False, frame_idx=fobs.frame_idx, conf=max(o.conf for o in spills))
            s["last"] = t
            if not s["flagged"] and t - s["first"] >= cfg.surface_limit_s:
                s["flagged"] = True
                self._emit(Finding("SURFACE_NOT_CLEANED", "violation", None, s["first"], t, round(min(1.0, s["conf"] + 0.1), 3),
                                   f"Spill on the prep counter not cleaned within {cfg.surface_limit_s:.0f}s",
                                   {"spill_first_seen": round(s["first"], 1)}, frame_idx=fobs.frame_idx))
        elif s is not None and t - s["last"] > 3.0:
            self._close_spill()

    def _close_spill(self):
        s, self.spill = self.spill, None
        if not s["flagged"]:
            self._emit(Finding("SURFACE_NOT_CLEANED", "compliant", None, s["first"], s["last"], 0.7,
                               f"Spill cleaned after {s['last'] - s['first']:.0f}s", {}, frame_idx=s["frame_idx"]))

    # ------------------------------------------------------------------ end of clip
    def finish(self):
        self._new = []
        for ps in self.people.values():
            if ps.rub:
                self._finish_rub(ps)
            for z in list(ps.open_visits):
                self._close_visit(ps, z)
            self._maybe_end_session(ps, self.t, force=True)
        for key in list(self.pests):
            self._close_pest(key)
        if self.spill is not None:
            self._close_spill()
        return self._new
