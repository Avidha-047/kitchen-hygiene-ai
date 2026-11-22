"""What the perception layer hands to the reasoning layer for one sampled frame.

Keeping this a plain data structure means the reasoning layer can be tested with
hand-written observations, without any video or model.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PersonObs:
    track_id: int
    box: tuple                      # x1, y1, x2, y2 in pixels
    conf: float
    wrists: list                    # [(x, y, conf), (x, y, conf)]  left, right
    lather: Optional[float] = None  # fraction of near-white pixels around the hands (filled by perception)

    @property
    def centre(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self):
        return max(1.0, self.box[2] - self.box[0])

    @property
    def height(self):
        return max(1.0, self.box[3] - self.box[1])

    def visible_wrists(self, min_conf):
        return [(x, y) for x, y, c in self.wrists if c >= min_conf]

    @property
    def head_box(self):
        x1, y1, x2, y2 = self.box
        return (x1, y1, x2, y1 + 0.22 * (y2 - y1))


@dataclass
class ObjObs:
    cls: str
    conf: float
    box: tuple

    @property
    def centre(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class FrameObs:
    t: float                        # seconds since the start of the clip
    frame_idx: int
    width: int
    height: int
    persons: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    image: Optional[np.ndarray] = None   # kept only for keyframes; dropped afterwards to save memory
