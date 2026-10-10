"""python -m src.film.make stills <shot> t1 t2 ...   |  shot <name> [...]  |  assemble  |  info"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from .core import FPS, OUT, H, W, VideoReader, VideoWriter, to_np8
from .timeline import build, schedule


def get(name):
    for shot, xf in build():
        if shot.name == name:
            return shot
    raise KeyError(name)


def stills(name, times, cols=2, scale=0.5):
    shot = get(name)
    shot.setup()
    os.makedirs(os.path.join(OUT, "preview"), exist_ok=True)
    tiles = []
    for t in times:
        t0 = time.time()
        f = to_np8(shot.frame(float(t)))
        torch.cuda.synchronize()
        print(f"{name} t={float(t):.2f}s rendered in {time.time()-t0:.2f}s")
        Image.fromarray(f).save(os.path.join(OUT, "preview", f"{name}_{float(t):06.2f}.png"))
        tiles.append(np.array(Image.fromarray(f).resize((int(W * scale), int(H * scale)), Image.LANCZOS)))
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, tl in enumerate(tiles):
        sheet[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = tl
    path = os.path.join(OUT, "preview", f"sheet_{name}.png")
    Image.fromarray(sheet).save(path)
    print("sheet ->", path)


def render_shot(name):
    shot = get(name)
    shot.setup()
    n = int(round(shot.duration * FPS))
    path = os.path.join(OUT, "shots", f"{name}.mp4")
    vw = VideoWriter(path + ".part.mp4")
    t0 = time.time()
    for k in range(n):
        vw.write(shot.frame(k / FPS))
        if k % 60 == 0:
            el = time.time() - t0
            print(f"{name}: frame {k}/{n}  {el:.0f}s elapsed, ETA {el / max(k, 1) * (n - k):.0f}s", flush=True)
    vw.close()
    os.replace(path + ".part.mp4", path)
    print(f"{name}: {n} frames in {time.time()-t0:.0f}s -> {path}")


def assemble(out_name="film_video.mp4"):
    """Crossfade all shot intermediates into one video stream (no audio)."""
    sched, total = schedule()
    n_total = int(round(total * FPS))
    readers = {}
    vw = VideoWriter(os.path.join(OUT, out_name), crf=14, preset="slow", pix="yuv420p")
    starts = [(s, int(round(t0 * FPS)), int(round(s.duration * FPS)), xf) for s, t0, xf in sched]
    for f in range(n_total):
        acc = None
        for k, (s, f0, nf, xf) in enumerate(starts):
            if f0 <= f < f0 + nf:
                if s.name not in readers:
                    readers[s.name] = VideoReader(os.path.join(OUT, "shots", f"{s.name}.mp4"))
                fr = readers[s.name].read()
                if fr is None:
                    fr = np.zeros((H, W, 3), np.uint8)
                fr = fr.astype(np.float32)
                # weight: fade in over the previous shot's crossfade, fade out over our own
                w = 1.0
                if k > 0:
                    xin = starts[k - 1][3]
                    if xin > 0 and f < f0 + xin * FPS:
                        w = (f - f0 + 0.5) / (xin * FPS)
                acc = fr * w if acc is None else acc * (1 - w) + fr * w
        vw.write(np.clip(acc + 0.5, 0, 255).astype(np.uint8) if acc is not None else np.zeros((H, W, 3), np.uint8))
        if f % 300 == 0:
            print(f"assemble {f}/{n_total}", flush=True)
    vw.close()
    for r in readers.values():
        r.close()
    print("assembled", total, "s")


def info():
    sched, total = schedule()
    for s, t0, xf in sched:
        print(f"{s.name:10s} start {t0:7.2f}s  dur {s.duration:6.2f}s  xfade->next {xf:.1f}s  cues: " +
              ", ".join(f"{c.seg}@{t0 + c.start:.1f}" for c in s.cues))
    print(f"total {total:.1f}s = {total/60:.2f} min")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stills":
        stills(sys.argv[2], sys.argv[3:])
    elif cmd == "shot":
        for nm in sys.argv[2:]:
            render_shot(nm)
    elif cmd == "assemble":
        assemble()
    elif cmd == "info":
        info()
