"""Convert a Pascal-VOC (xml) image folder to a YOLO dataset with train/val/test splits."""
import random, shutil, sys
import xml.etree.ElementTree as ET
from pathlib import Path

src_train, src_test, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
classes = ["cockroach"]


def convert(xml_path, img_dst, lbl_dst):
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in classes:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(f"{classes.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    img = xml_path.with_suffix(".jpg")
    if not img.exists() or not lines or w <= 0 or h <= 0:
        return 0
    shutil.copy(img, img_dst / img.name)
    (lbl_dst / (img.stem + ".txt")).write_text("\n".join(lines) + "\n")
    return len(lines)


train_xml = sorted(src_train.glob("*.xml"))
random.Random(7).shuffle(train_xml)
n_val = int(0.15 * len(train_xml))
splits = {"valid": train_xml[:n_val], "train": train_xml[n_val:], "test": sorted(src_test.glob("*.xml"))}
stats = {}
for split, files in splits.items():
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    imgs = boxes = 0
    for x in files:
        k = convert(x, out / split / "images", out / split / "labels")
        imgs += 1 if k else 0
        boxes += k
    stats[split] = (imgs, boxes)
(out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: cockroach\n")
print(stats)
