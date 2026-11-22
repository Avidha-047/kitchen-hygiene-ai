"""Build the 3–5 minute scenario video (deliverable 2) from the scene clips.

For every scene: a title card (what happens + the expected result), the original clip,
then the same clip as the system saw it (zones, person ID, wrists, alerts), if it exists.

Usage:
  python tools/build_scenario_video.py VIDEOS_DIR OUTPUT_DIR OUT.mp4 [narrate=yes|no]
  Sound: each clip keeps its own sound (Veo makes kitchen sounds); title cards get a spoken
  narration (pip install gtts; needs internet). Without gtts the cards are silent.
  VIDEOS_DIR  folder with S01.mp4, S02.mp4 ...
  OUTPUT_DIR  the notebook's output folder (contains evidence/<scene>/<scene>_annotated.mp4); may be empty
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 24
SCENES = {
    "S01": ("Proper hand wash with soap, then food prep", "Expected: compliant"),
    "S02": ("Cook B goes straight to the counter", "Expected: hand-wash violation"),
    "S03": ("Hands rubbed with water only, no soap", "Expected: hand-wash violation (no soap)"),
    "S04": ("Carrots rinsed at the sink before chopping", "Expected: vegetables compliant"),
    "S05": ("Carrots chopped without washing them", "Expected: vegetable-wash violation"),
    "S06": ("Clean ladle from the top shelf", "Expected: compliant"),
    "S07": ("Dirty ladle from the used tray put into the pot", "Expected: dirty-utensil violation"),
    "S08": ("Raw chicken, hand wash, then cooked rice", "Expected: compliant"),
    "S09": ("Raw chicken, then cooked rice without washing", "Expected: cross-contamination risk"),
    "S10": ("Cook wearing a cap and gloves", "Expected: compliant"),
    "S11": ("Cook without a head cover", "Expected: head-cover violation"),
    "S12": ("A cockroach on the counter", "Expected: pest alert"),
    "S13": ("A rat on the floor", "Expected: pest alert"),
    "S14": ("Spill wiped straight away", "Expected: compliant"),
    "S15": ("Spill left on the counter", "Expected: surface not cleaned"),
    "S16": ("Oil from the bottle marked USED", "Expected: oil flag"),
    "H1": ("Hard case: two cooks, only one washes", "Expected: only Cook B flagged"),
    "H2": ("Hard case: someone blocks the view of the sink", "Expected: uncertain, not a violation"),
    "H3": ("Hard case: dim light", "Expected: compliant"),
    "H4": ("Hard case: dark vegetable that looks like a cockroach", "Expected: no alert"),
    "H5": ("Hard case: a fly for about one second", "Expected: pest alert"),
    "H6": ("Hard case: phone touched after washing", "Expected: hand-wash violation"),
}
BG, TEAL, CORAL, INK = (234, 241, 244), (107, 125, 47), (28, 68, 194), (29, 33, 22)


def card(lines, path):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 14), TEAL, -1)
    y = 250
    for i, (text, scale, col, thick) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        cv2.putText(img, text, ((W - tw) // 2, y), cv2.FONT_HERSHEY_DUPLEX, scale, col, thick, cv2.LINE_AA)
        y += int(th * 2.2)
    cv2.imwrite(str(path), img)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])


AR = 44100


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def speak(text, out_mp3):
    """Narration with Google text-to-speech (Indian English voice). Returns the mp3 path, or None if unavailable."""
    try:
        from gtts import gTTS
        gTTS(text, lang="en", tld="co.in").save(str(out_mp3))
        return out_mp3
    except Exception as e:                       # no internet / gtts missing: the video is still built, silently
        print("  (narration skipped:", str(e)[:80], ")")
        return None


def still(png, secs, out, voice=None):
    """A title card. With a voice file the card lasts as long as the narration (at least `secs`)."""
    if voice:
        secs = max(secs, duration(voice) + 0.4)
        audio = ["-i", str(voice), "-filter_complex", f"[1:a]apad,atrim=0:{secs},aresample={AR}[a]", "-map", "0:v", "-map", "[a]"]
    else:
        audio = ["-f", "lavfi", "-i", f"anullsrc=r={AR}:cl=stereo", "-map", "0:v", "-map", "1:a"]
    run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), *audio, "-t", str(secs),
         "-vf", f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
         "-c:a", "aac", "-ar", str(AR), "-ac", "2", "-shortest", str(out)])


def norm(src, out, label=None, sound_from=None):
    """Resize to 1280x720 at 24 fps. Sound: the clip's own audio (or `sound_from`'s), else silence."""
    vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},setsar=1,format=yuv420p"
    snd = sound_from if sound_from is not None else src
    d = duration(src)
    if has_audio(snd):
        audio = ["-i", str(snd), "-map", "0:v", "-map", "1:a", "-af", f"apad,atrim=0:{d},aresample={AR}"]
    else:
        audio = ["-f", "lavfi", "-i", f"anullsrc=r={AR}:cl=stereo", "-map", "0:v", "-map", "1:a"]
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *audio, "-vf", vf, "-t", str(d),
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-ar", str(AR), "-ac", "2", str(out)])
    if label:   # stamp a small label in the corner, frame by frame with OpenCV (no font files needed)
        cap = cv2.VideoCapture(str(out)); frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            cv2.rectangle(f, (0, H - 46), (W, H), (20, 20, 20), -1)
            cv2.putText(f, label, (20, H - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(f)
        cap.release()
        tmp = out.with_name(out.stem + "_raw.mp4")
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
        done = out.with_name(out.stem + "_lab.mp4")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(out), "-map", "0:v", "-map", "1:a",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "copy", str(done)])
        tmp.unlink()
        done.replace(out)


def main(videos, outputs, dest, narrate="yes"):
    videos, outputs, dest = Path(videos), Path(outputs), Path(dest)
    talk = narrate.lower() not in ("no", "0", "false")
    order = [s for s in SCENES if (videos / f"{s}.mp4").exists()]
    tmp = Path(tempfile.mkdtemp())
    parts = []
    c = tmp / "intro.png"
    card([("Can AI Be a Kitchen Hygiene Inspector?", 1.5, TEAL, 2), ("Scenario video", 1.0, INK, 1),
          (f"{len(order)} scenes: compliant behaviour, violations and hard cases", 0.8, INK, 1),
          ("All scenes are AI-generated test footage (Google Veo), from one fixed CCTV view", 0.7, CORAL, 1)], c)
    v = speak("Scenario video. Every scene is AI-generated, from one fixed camera. "
              "You will see each clip, then what our system saw.", tmp / "intro.mp3") if talk else None
    still(c, 5, tmp / "000_intro.mp4", v); parts.append(tmp / "000_intro.mp4")
    for i, s in enumerate(order, 1):
        what, expected = SCENES[s]
        c = tmp / f"{s}.png"
        card([(f"Scene {s}", 1.6, TEAL, 2), (what, 1.0, INK, 1), (expected, 0.9, CORAL, 1)], c)
        name = (f"Scene {int(s[1:])}" if s[0] == "S" else f"Hard case {int(s[1:])}") if s[1:].isdigit() else s
        v = speak(f"{name}. {what}.", tmp / f"{s}.mp3") if talk else None
        still(c, 3, tmp / f"{i:03d}_a.mp4", v); parts.append(tmp / f"{i:03d}_a.mp4")
        norm(videos / f"{s}.mp4", tmp / f"{i:03d}_b.mp4", f"{s} - original clip")
        parts.append(tmp / f"{i:03d}_b.mp4")
        ann = outputs / "evidence" / s / f"{s}_annotated.mp4"
        if ann.exists():
            norm(ann, tmp / f"{i:03d}_c.mp4", f"{s} - what the system saw (zones, person ID, wrists, alerts)",
                 sound_from=videos / f"{s}.mp4")        # same timing, so reuse the clip's own sound
            parts.append(tmp / f"{i:03d}_c.mp4")
        print(f"{s}: added" + (" with annotated version" if ann.exists() else ""))
    c = tmp / "end.png"
    card([("AI flags. Evidence decides.", 1.4, TEAL, 2), ("Avidha Chandra", 0.9, INK, 1)], c)
    v = speak("AI flags. Evidence decides. Thank you.", tmp / "end.mp3") if talk else None
    still(c, 4, tmp / "999_end.mp4", v); parts.append(tmp / "999_end.mp4")
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dest)])
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dest)],
                             capture_output=True, text=True).stdout.strip())
    print(f"\nSaved {dest}  ({int(d // 60)} min {int(d % 60)} s)")
    if d > 300:
        print("Over 5 minutes: run again with narration off (add  no  at the end of the command).")


if __name__ == "__main__":
    main(*sys.argv[1:5])
