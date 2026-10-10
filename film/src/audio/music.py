"""Procedural ambient score, laid out against the film's timeline.

Ingredients (all numpy): additive-synthesis pads (soft triangle-like spectra, detuned
chorus, slow cross-fades), a drone and sub, plucked arpeggios through a stereo delay,
high shimmer partials, and a synthetic convolution reverb.  Harmony stays in D minor
(i-VI-III-VII colours) and resolves to D major when the ring appears.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
SR = 48000
RNG = np.random.default_rng(2026)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


CHORDS = {
    "Dm9": [50, 57, 64, 65],
    "Bbmaj7": [46, 53, 57, 62],
    "Fadd9": [41, 57, 60, 67],
    "Csus2": [48, 55, 62, 64],
    "Gm9": [43, 58, 62, 65],
    "Asus4": [45, 57, 62, 64],
    "A": [45, 57, 61, 64],
    "Dmaj": [38, 50, 57, 62, 66],
    "Dm": [38, 50, 57, 62, 65],
}


def env_adsr(n, a, r, sr=SR):
    """Raised-cosine attack and release (no clicks)."""
    e = np.ones(n)
    na, nr = min(int(a * sr), n // 2), min(int(r * sr), n // 2)
    if na > 0:
        e[:na] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, na))
    if nr > 0:
        e[-nr:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nr))
    return e


def pad_note(freq, dur, amp=1.0, attack=3.0, release=3.5, bright=0.5, detune_cents=6.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for k, (dc, pan) in enumerate(((-detune_cents, 0.2), (0.0, 0.5), (detune_cents, 0.8))):
        f0 = freq * 2 ** (dc / 1200)
        sig = np.zeros(n)
        vib = 1 + 0.0012 * np.sin(2 * np.pi * (0.13 + 0.05 * k) * t + RNG.uniform(0, 6.28))
        ph0 = RNG.uniform(0, 2 * np.pi, 16)
        for h in range(1, 12):
            fh = f0 * h
            if fh > 9000:
                break
            a_h = (1.0 / h ** 2) * (1.0 if h % 2 else 0.55 * bright) * math.exp(-(h - 1) * (0.55 - 0.4 * bright))
            sig += a_h * np.sin(2 * np.pi * fh * t * vib + ph0[h])
        out[0] += sig * math.cos(pan * math.pi / 2)
        out[1] += sig * math.sin(pan * math.pi / 2)
    return out * env_adsr(n, attack, release) * amp / 3


def pluck(freq, dur=2.4, amp=1.0, pan=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t) * np.exp(-t * 6)
           + 0.08 * np.sin(6 * np.pi * freq * t) * np.exp(-t * 9))
    e = np.exp(-t * 2.2) * (1 - np.exp(-t * 400))
    s = sig * e * amp
    return np.stack([s * math.cos(pan * math.pi / 2), s * math.sin(pan * math.pi / 2)])


def add(buf, sig, t0):
    i0 = int(t0 * SR)
    if i0 >= buf.shape[1]:
        return
    n = min(sig.shape[1], buf.shape[1] - i0)
    if i0 < 0:
        sig = sig[:, -i0:]
        n = min(sig.shape[1], buf.shape[1])
        i0 = 0
    buf[:, i0:i0 + n] += sig[:, :n]


def reverb_ir(rt60=4.5, dur=6.0, predelay=0.03):
    n = int(dur * SR)
    t = np.arange(n) / SR
    decay = np.exp(-6.91 * t / rt60)
    ir = RNG.normal(size=(2, n)) * decay
    # darken the tail: progressively low-passed copies blended by time
    sos = butter(2, 3000, fs=SR, output="sos")
    dark = sosfilt(sos, ir, axis=1)
    w = np.clip(t / 1.5, 0, 1)
    ir = ir * (1 - w) + dark * w
    ir[:, : int(predelay * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return ir


def stereo_delay(x, delay_s=0.42, fb=0.38, mix=0.35):
    d = int(delay_s * SR)
    y = x.copy()
    for k in range(1, 6):
        g = fb ** k
        sh = d * k
        if sh >= x.shape[1]:
            break
        src = x[::-1] if k % 2 else x  # ping-pong: swap channels on odd taps
        y[:, sh:] += mix * g * src[:, : x.shape[1] - sh]
    return y


def build(total, shots):
    """shots: dict name -> (start, duration).  Returns (2, N) float array."""
    N = int(total * SR) + SR * 2
    pads = np.zeros((2, N))
    plk = np.zeros((2, N))
    low = np.zeros((2, N))
    shim = np.zeros((2, N))
    S = lambda nm: shots[nm][0]
    E = lambda nm: shots[nm][0] + shots[nm][1]

    def chord_seq(t0, t1, names, amp=0.5, bright=0.45, overlap=3.5):
        seg = (t1 - t0) / len(names)
        for k, nm in enumerate(names):
            a = t0 + k * seg
            for m in CHORDS[nm]:
                add(pads, pad_note(midi_hz(m), seg + overlap, amp * (0.8 if m < 48 else 1.0), attack=overlap * 0.9,
                                   release=overlap, bright=bright), a)

    # --- harmonic plan, section by section
    chord_seq(S("open") + 0.5, E("open") - 6.0, ["Dm9", "Bbmaj7"], amp=0.42, bright=0.35, overlap=5.0)
    chord_seq(E("open") - 7.0, E("open"), ["Dm9"], amp=0.55, bright=0.6, overlap=4.0)  # title swell
    chord_seq(S("lensing"), E("lensing"), ["Bbmaj7", "Fadd9", "Csus2"], amp=0.42, bright=0.4)
    chord_seq(S("rays"), E("rays"), ["Dm9", "Bbmaj7", "Gm9", "Asus4"], amp=0.38, bright=0.4)
    chord_seq(S("disk"), E("doppler"), ["Dm9", "Fadd9", "Bbmaj7", "Csus2", "Dm9", "Gm9"], amp=0.44, bright=0.5)
    chord_seq(S("spin"), E("spin"), ["Bbmaj7", "Csus2", "A"], amp=0.48, bright=0.6)
    chord_seq(S("scale"), E("scale"), ["Dm9", "Bbmaj7", "Fadd9", "Csus2"], amp=0.34, bright=0.3)
    chord_seq(S("eht"), E("eht"), ["Gm9", "Bbmaj7", "Csus2", "Dm9"], amp=0.38, bright=0.45)
    chord_seq(S("fourier"), E("fourier"), ["Dm9", "Fadd9", "Gm9", "Asus4"], amp=0.36, bright=0.45)
    rec = shots["recon"]
    t_ring = shots["_ring"]
    chord_seq(S("recon"), t_ring - 0.5, ["Bbmaj7", "Gm9", "Asus4", "A"], amp=0.40, bright=0.5)
    chord_seq(t_ring - 0.8, E("finale") - 6, ["Dmaj", "Bbmaj7", "Dmaj"], amp=0.5, bright=0.65, overlap=4.0)
    chord_seq(E("finale") - 8, E("credits"), ["Dm9", "Dmaj"], amp=0.36, bright=0.35, overlap=6.0)

    # --- drone and sub (D), breathing slowly; absent in the "scale" section for space
    t = np.arange(N) / SR
    drone = 0.20 * np.sin(2 * np.pi * midi_hz(38) * t) + 0.10 * np.sin(2 * np.pi * midi_hz(26) * t)
    drone *= 0.75 + 0.25 * np.sin(2 * np.pi * t / 23.0)
    g = np.ones(N)
    for a_, b_, lvl in ((S("scale"), E("scale"), 0.25), (S("fourier"), E("fourier"), 0.6)):
        i0, i1 = int(a_ * SR), int(b_ * SR)
        g[i0:i1] = lvl
    win = np.hanning(SR * 4)
    g = fftconvolve(g, win / win.sum(), mode="same")  # (a direct np.convolve here is ~1e12 ops)
    g[: int(1.0 * SR)] *= np.linspace(0, 1, int(1.0 * SR))
    low[0] += drone * g
    low[1] += drone * g

    # --- plucked arpeggios: rays (following light), EHT (data), Fourier (ripples)
    def arp(t0, t1, names, step, amp, octave=12):
        k = 0
        tt = t0
        while tt < t1:
            nm = names[min(int((tt - t0) / (t1 - t0) * len(names)), len(names) - 1)]
            notes = sorted(CHORDS[nm])[1:]
            m = notes[k % len(notes)] + octave
            fade = min(1.0, (tt - t0) / 3.0, (t1 - tt) / 3.0)
            add(plk, pluck(midi_hz(m), amp=amp * fade * (0.85 + 0.3 * RNG.random()), pan=0.25 + 0.5 * RNG.random()), tt)
            k += 1
            tt += step
    arp(S("rays") + 2, E("rays") - 2, ["Dm9", "Bbmaj7", "Gm9", "Asus4"], 0.84, 0.06)
    arp(S("eht") + 4, E("eht") - 1, ["Gm9", "Bbmaj7", "Csus2", "Dm9"], 0.42, 0.045, octave=24)
    arp(S("fourier") + 1, E("fourier") - 2, ["Dm9", "Fadd9", "Gm9", "Asus4"], 0.56, 0.05)

    # --- shimmer: high octave partials for the title, the scale section, and the ring
    def shimmer(t0, t1, notes, amp):
        n = int((t1 - t0) * SR)
        tt = np.arange(n) / SR
        s = np.zeros((2, n))
        for k, m in enumerate(notes):
            f = midi_hz(m)
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * (0.07 + 0.031 * k) * tt + RNG.uniform(0, 6.28))
            sig = np.sin(2 * np.pi * f * tt) * lfo
            p = (k % 3) / 2
            s[0] += sig * math.cos(p * math.pi / 2)
            s[1] += sig * math.sin(p * math.pi / 2)
        add(shim, s * env_adsr(n, 3.0, 4.0) * amp / len(notes), t0)
    shimmer(E("open") - 8, E("open") + 2, [74, 81, 86, 88], 0.05)
    shimmer(S("scale") + 4, S("scale") + 24, [81, 86, 88, 93], 0.035)
    shimmer(t_ring - 0.5, t_ring + 12, [74, 78, 81, 86, 90], 0.06)

    # --- effects and mix
    plk = stereo_delay(plk)
    dry = pads + low * 0.9 + plk + shim
    ir = reverb_ir()
    wet = np.stack([fftconvolve(dry[c] * (0.6 if c == 0 else 0.6), ir[c])[:N] for c in range(2)])
    mix = 0.75 * dry + 0.55 * wet
    # gentle high-cut and DC/rumble removal
    mix = sosfilt(butter(2, 30, "highpass", fs=SR, output="sos"), mix, axis=1)
    mix = sosfilt(butter(2, 9000, "lowpass", fs=SR, output="sos"), mix, axis=1)
    # global arc: fade in at the start, fade out at the end
    n_in, n_out = int(2.5 * SR), int(6.0 * SR)
    mix[:, :n_in] *= np.linspace(0, 1, n_in) ** 2
    end = int(total * SR)
    mix[:, end - n_out:end] *= np.linspace(1, 0, n_out) ** 1.5
    mix[:, end:] = 0
    return mix[:, :end] / (np.abs(mix).max() + 1e-9) * 0.9


def timeline_info():
    from src.film.timeline import schedule
    sched, total = schedule()
    shots = {s.name: (t0, s.duration) for s, t0, xf in sched}
    rec = [s for s, t0, xf in sched if s.name == "recon"][0]
    shots["_ring"] = shots["recon"][0] + rec.t_ring
    return total, shots, sched


if __name__ == "__main__":
    import soundfile as sf
    total, shots, _ = timeline_info()
    m = build(total, shots)
    os.makedirs(os.path.join(ROOT, "out", "audio"), exist_ok=True)
    sf.write(os.path.join(ROOT, "out", "audio", "music.wav"), m.T.astype(np.float32), SR)
    print(f"music: {m.shape[1] / SR:.1f}s, peak {np.abs(m).max():.3f}")
