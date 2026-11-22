"""Join AI-generated scene parts (S01_a.mp4, S01_b.mp4 ...) into one clip per scene (S01.mp4).

Parts are moved into videos/parts/ afterwards so the pipeline only sees whole scenes.
Clips are re-encoded to one size and frame rate, because different AI tools export differently.

Usage:  python tools/join_parts.py /path/to/videos
"""
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

folder = Path(sys.argv[1])
parts_dir = folder / "parts"
parts_dir.mkdir(exist_ok=True)
groups = defaultdict(list)
# fix double extensions like S01_a.mp4.mp4 (Windows hides ".mp4", so people type it twice)
for p in sorted(folder.iterdir()):
    fixed = re.sub(r"(\.mp4)+$", ".mp4", p.name, flags=re.I)
    if fixed != p.name and not (folder / fixed).exists():
        p.rename(folder / fixed)
        print(f"renamed {p.name} -> {fixed}")
for p in sorted(folder.iterdir()):
    m = re.match(r"^([A-Za-z]+\d+)_([a-z])\.(mp4|mov|webm|mkv)$", p.name)
    if m:
        groups[m.group(1)].append(p)

for scene, files in sorted(groups.items()):
    files = sorted(files)
    out = folder / f"{scene}.mp4"
    inputs = []
    for f in files:
        inputs += ["-i", str(f)]
    n = len(files)
    filt = "".join(f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=24,setsar=1[v{i}];" for i in range(n))
    filt += "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", filt, "-map", "[out]",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{scene}: FAILED - {r.stderr.strip()[:200]}")
        continue
    for f in files:
        shutil.move(str(f), parts_dir / f.name)
    print(f"{scene}: joined {', '.join(f.name for f in files)} -> {out.name}")

print("whole scenes now in videos/:", sorted(p.name for p in folder.iterdir() if p.suffix.lower() in {'.mp4', '.mov', '.webm', '.mkv'}))
