from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


DATASET = Path.home() / "datasets/Dataset_UAV123_10fps/UAV123_10fps"
FRAMES = DATASET / "data_seq/UAV123_10fps"
ANNOTATIONS = DATASET / "anno/UAV123_10fps"
CLASSES = sorted({"bird", "bike", "boat", "building", "car", "group", "person", "truck", "uav", "wakeboard"})


@dataclass
class Sequence:
    name: str
    folder: str
    start: int
    end: int
    boxes: list[tuple[float, float, float, float] | None]


def sequences() -> list[Sequence]:
    config = (DATASET / "configSeqs.m").read_text()
    pattern = re.compile(r"struct\('name','([^']+)','path','[^']*\\([^\\]+)\\','startFrame',(\d+),'endFrame',(\d+)")
    result = []
    for name, folder, start, end in pattern.findall(config):
        annotation = ANNOTATIONS / f"{name}.txt"
        if not annotation.exists():
            continue
        boxes = []
        for row in annotation.read_text().splitlines():
            try:
                box = tuple(map(float, row.split(",")))
                boxes.append(box if len(box) == 4 and all(map(math.isfinite, box)) else None)
            except ValueError:
                boxes.append(None)
        result.append(Sequence(name, folder, int(start), int(end), boxes))
    return result


def crop_target(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    x, y, width, height = box
    side = max(width, height) * 1.25
    cx, cy = x + width / 2, y + height / 2
    crop = image.crop((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    return ImageOps.pad(crop, (224, 224), method=Image.Resampling.BILINEAR, color=(114, 114, 114))


def box_for_frame(records: list[Sequence], frame_number: int) -> tuple[str, tuple[float, float, float, float] | None] | None:
    for record in records:
        if record.start <= frame_number <= record.end:
            index = frame_number - record.start
            if index < len(record.boxes):
                return record.name.split("_")[0].rstrip("0123456789"), record.boxes[index]
    return None
