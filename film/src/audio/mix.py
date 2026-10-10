"""Place narration on the timeline, duck the music under it, normalise loudness, write subtitles.

Outputs: out/audio/narration_track.wav, out/audio/mix.wav (48 kHz stereo), out/film.srt
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.audio.music import SR, timeline_info  # noqa: E402
from script.narration import SEGMENTS  # noqa: E402

NARR = os.path.join(ROOT, "out", "audio", "narration")


def lufs(x, sr=SR):
    import pyloudnorm as pyln
    return pyln.Meter(sr).integrated_loudness(x.T if x.ndim == 2 else x)


def compress(x, thresh_db=-24.0, ratio=3.0, att=0.010, rel=0.150):
    """Feed-forward RMS compressor (control rate 1 ms)."""
    hop = SR // 1000
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    lvl = 20 * np.log10(rms)
    over = np.maximum(lvl - thresh_db, 0)
    target = -over * (1 - 1 / ratio)
    g = np.zeros(n)
    y = 0.0
    aa, ar = np.exp(-1e-3 / att), np.exp(-1e-3 / rel)
    for i in range(n):
        a = aa if target[i] < y else ar
        y = a * y + (1 - a) * target[i]
        g[i] = y
    gain = np.repeat(10 ** (g / 20), hop)
    gain = np.pad(gain, (0, len(x) - len(gain)), mode="edge")
    return x * gain


def limiter_gain(x, ceiling=0.84, look=0.005, release=0.08):
    """Look-ahead brick-wall limiter gain: min-filter of the required gain over 2L, then a
    length-L moving average (which never exceeds the requirement), then a smooth release."""
    from scipy.ndimage import minimum_filter1d
    peak = np.abs(x).max(axis=0)
    req = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    L = int(look * SR)
    g = minimum_filter1d(req, size=2 * L + 1)
    cs = np.concatenate([[0.0], np.cumsum(g)])
    g = np.pad((cs[L:] - cs[:-L]) / L, (L // 2, L - L // 2 - 1), mode="edge")
    g = np.minimum(g, minimum_filter1d(req, size=2 * L + 1))
    # release smoothing (instant attack, exponential recovery), control rate
    hop = 48
    out = g.copy()
    ar = np.exp(-hop / (release * SR))
    y = 1.0
    for i in range(0, len(g), hop):
        m = g[i:i + hop].min()
        y = m if m < y else ar * y + (1 - ar) * m
        out[i:i + hop] = np.minimum(g[i:i + hop], y)
    return out


def srt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def split_subtitle(text, words, t0, max_chars=84):
    """Split a segment's subtitle into chunks of <= 2 lines, timed by its word timestamps."""
    parts = re.split(r"(?<=[.!?:;])\s+", text)
    chunks = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(", ", 0, max_chars)
            cut = cut + 1 if cut > 20 else p.rfind(" ", 0, max_chars)
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            chunks.append(p)
    # distribute time by character share, snapped to word starts
    total = sum(len(c) for c in chunks)
    wt = [w["t0"] for w in words] + [words[-1]["t1"] if words else 0]
    dur = wt[-1]
    out, acc = [], 0
    for c in chunks:
        a = acc / total * dur
        acc += len(c)
        b = acc / total * dur
        sa = min(wt, key=lambda x: abs(x - a))
        sb = min(wt, key=lambda x: abs(x - b)) if acc < total else dur
        out.append((t0 + sa, t0 + max(sb, sa + 0.8), c))
    return out


def wrap2(s, width=42):
    if len(s) <= width:
        return s
    mid = len(s) // 2
    cut = min((i for i, ch in enumerate(s) if ch == " "), key=lambda i: abs(i - mid))
    return s[:cut] + "\n" + s[cut + 1:]


def main():
    total, shots, sched = timeline_info()
    meta = json.load(open(os.path.join(NARR, "narration.json")))
    N = int(total * SR)
    narr = np.zeros(N)
    subs = []
    seg_text = {sid: sub for sid, _, sub in SEGMENTS}
    for shot, t0, xf in sched:
        for c in shot.cues:
            wav, sr = sf.read(os.path.join(NARR, f"{c.seg}.wav"))
            wav = resample_poly(wav, SR, sr)
            i0 = int((t0 + c.start) * SR)
            n = min(len(wav), N - i0)
            narr[i0:i0 + n] += wav[:n]
            subs += split_subtitle(seg_text[c.seg], meta[c.seg]["words"], t0 + c.start)
    # gentle compression of the narration (3:1 above -24 dBFS RMS, 10 ms / 150 ms), then -16 LUFS
    narr = compress(narr)
    g = 10 ** ((-16.0 - lufs(narr)) / 20)
    narr *= g
    music, _ = sf.read(os.path.join(ROOT, "out", "audio", "music.wav"))
    music = music.T[:, :N]
    if music.shape[1] < N:
        music = np.pad(music, ((0, 0), (0, N - music.shape[1])))
    # sidechain ducking from the narration envelope
    env = np.abs(narr)
    k = int(0.04 * SR)
    cs = np.concatenate([[0.0], np.cumsum(env)])
    env = np.pad((cs[k:] - cs[:-k]) / k, (k // 2, k - k // 2 - 1), mode="edge")  # moving average
    active = (env > 10 ** (-40 / 20)).astype(float)
    # attack 80 ms, release 900 ms one-pole smoothing of the gate
    duck = np.zeros(N)
    a_att, a_rel = np.exp(-1 / (0.08 * SR)), np.exp(-1 / (0.9 * SR))
    y = 0.0
    for i in range(0, N, 64):  # control rate
        x = active[i]
        a = a_att if x > y else a_rel
        y = a ** 64 * y + (1 - a ** 64) * x
        duck[i:i + 64] = y
    music_gain_free = 10 ** ((-19.0 - lufs(music)) / 20)  # music level in the gaps
    duck_db = -16.0
    mg = music_gain_free * 10 ** (duck_db * duck / 20)
    music_stem = music * mg
    mix = music_stem + narr[None, :]
    # loudness to -16 LUFS integrated, then a look-ahead limiter at -1.5 dBFS
    gl = 10 ** ((-16.0 - lufs(mix)) / 20)
    mix *= gl
    lim = limiter_gain(mix, ceiling=10 ** (-1.5 / 20))
    mix *= lim
    out = os.path.join(ROOT, "out", "audio")
    sf.write(os.path.join(out, "narration_track.wav"), (narr * gl * lim).astype(np.float32), SR)
    sf.write(os.path.join(out, "music_stem.wav"), (music_stem * gl * lim).T.astype(np.float32), SR)
    sf.write(os.path.join(out, "mix.wav"), mix.T.astype(np.float32), SR)
    with open(os.path.join(ROOT, "out", "film.srt"), "w") as fh:
        for k, (a, b, txt) in enumerate(subs, 1):
            fh.write(f"{k}\n{srt_time(a)} --> {srt_time(b)}\n{wrap2(txt)}\n\n")
    print(f"mix: {total:.1f}s, integrated {lufs(mix):.1f} LUFS, peak {20 * np.log10(np.abs(mix).max()):.1f} dBFS, "
          f"narration alone {lufs(narr * gl):.1f} LUFS, {len(subs)} subtitles")


if __name__ == "__main__":
    main()
