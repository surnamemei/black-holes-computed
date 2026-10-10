"""Second stage: time-slide background, Q-transform, audio and page data for GW150914.

Uses the best template found by analysis.py (out/summary.json).
Run:  .venv/bin/python products.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import time

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d

from analysis import FHI, FLO, FS, GPS0, OUT, T_EVENT, Filter, chirp_mass, load, psd_welch, segment, template_fd

summ = json.load(open(os.path.join(OUT, "summary.json")))
M1, M2 = summ["best_m1"], summ["best_m2"]


def snr_series_full(x, f_psd, S, h_fn, win=64.0, keep=48.0):
    """|z(t)| over the whole file in overlapping windows (NaN where data are missing)."""
    N = len(x)
    out = np.full(N, np.nan)
    step = keep
    pad = (win - keep) / 2
    for c in np.arange(GPS0 + win / 2, GPS0 + 4096 - win / 2 + 1e-9, step):
        seg, t0 = segment(x, c, win)
        if len(seg) < int(win * FS) or not np.all(np.isfinite(seg)):
            continue
        F = Filter(seg, t0, f_psd, S)
        z, _ = F.snr(h_fn(F.f))
        i0 = int((t0 - GPS0) * FS)
        a, b = int(pad * FS), int((pad + keep) * FS)
        out[i0 + a:i0 + b] = np.abs(z[a:b])
    return out


def main():
    t0 = time.time()
    res = {}
    x, psd = {}, {}
    for det in ("H1", "L1"):
        x[det] = load(det)
        psd[det] = psd_welch(x[det])[:2]  # (f, S)
    hfn = lambda f: template_fd(M1, M2, f)
    rho = {d: snr_series_full(x[d], *psd[d], hfn) for d in ("H1", "L1")}
    # ------------------------------------------------ time slides of the network statistic
    w = int(0.010 * FS)
    LmaxW = maximum_filter1d(np.nan_to_num(rho["L1"], nan=0.0), size=2 * w + 1, mode="nearest")
    validL = maximum_filter1d(np.isfinite(rho["L1"]).astype(np.uint8), size=2 * w + 1) > 0
    H = np.nan_to_num(rho["H1"], nan=0.0)
    tt = GPS0 + np.arange(len(H)) / FS
    near = np.abs(tt - T_EVENT) < 1.0  # remove the real event from H1 times
    slides = [s for s in range(-1500, 1501, 3) if s != 0]
    loudest = []
    live = 0.0
    shift_samples = lambda s: s * FS
    for s in slides:
        L = np.roll(LmaxW, -shift_samples(s))
        vL = np.roll(validL, -shift_samples(s))
        L_near = np.roll(near, -shift_samples(s))  # the event in L1 time, wrapped like the data
        ok = np.isfinite(rho["H1"]) & vL & ~near & ~L_near
        net = np.where(ok, np.sqrt(H ** 2 + L ** 2), 0.0)
        loudest.append(float(net.max()))
        live += ok.sum() / FS
    loudest = np.array(loudest)
    res["slides"] = len(slides)
    res["background_live_days"] = live / 86400
    res["background_loudest"] = loudest.round(3).tolist()
    res["background_max"] = float(loudest.max())
    res["observed_network"] = summ["snr_network"]
    print(f"time slides: {len(slides)} shifts (3 s steps), {live / 86400:.1f} days of background live time; "
          f"loudest background network SNR {loudest.max():.2f}; observed {summ['snr_network']:.2f}")
    print(f"  => false-alarm rate < 1 per {live / 86400:.0f} days for this simple statistic (no chi^2, no glitch vetoes)")
    # ------------------------------------------------ products around the event
    seg = {}
    F = {}
    for det in ("H1", "L1"):
        s_, t0_ = segment(x[det], T_EVENT, 32.0)
        F[det] = Filter(s_, t0_, *psd[det])
        seg[det] = (s_, t0_)
    tmpl = np.load(os.path.join(OUT, "templates.npz"))
    w_t = lambda det: F[det].whitened()
    t_axis = lambda det: F[det].t0 + np.arange(F[det].N) / FS - T_EVENT
    def window(arr, det, a=-0.30, b=0.06, dec=1):
        tax = t_axis(det)
        m = (tax >= a) & (tax <= b)
        return tax[m][::dec], arr[m][::dec]
    tH, wH = window(w_t("H1"), "H1")
    _, wL = window(w_t("L1"), "L1")
    _, mH = window(tmpl["H1"], "H1")
    _, mL = window(tmpl["L1"], "L1")
    # raw strain (to show why whitening matters), 2 s, decimated
    tr, rawH = window(seg["H1"][0], "H1", -1.0, 0.2, 4)
    # Q-transform of whitened H1 (Gaussian tiles in frequency, Q = 8)
    def qtransform(det, Q=8.0, fmin=25, fmax=400, nf=120, a=-0.6, b=0.2):
        wf = np.fft.rfft(F[det].whitened(20, 1000))
        f = np.fft.rfftfreq(F[det].N, 1 / FS)
        tax = t_axis(det)
        m = (tax >= a) & (tax <= b)
        freqs = np.geomspace(fmin, fmax, nf)
        img = []
        for fc in freqs:
            bw = fc / Q
            g = np.exp(-0.5 * ((f - fc) / (bw / 2.355)) ** 2)
            z = np.fft.ifft(np.concatenate([wf * g, np.zeros(F[det].N - len(f))]))
            e = np.abs(z[m]) ** 2
            img.append(e)
        img = np.array(img)
        img = img / np.median(img)
        dec = max(1, img.shape[1] // 600)
        return tax[m][::dec], freqs, img[:, ::dec]
    qt, qf, qimg = qtransform("H1")
    # SNR time series around the event
    zH = np.load(os.path.join(OUT, "products.npz"))["zH"]
    zL = np.load(os.path.join(OUT, "products.npz"))["zL"]
    ts, sH = window(np.abs(zH), "H1", -0.25, 0.25, 4)
    _, sL = window(np.abs(zL), "L1", -0.25, 0.25, 4)
    # ASD + template characteristic strain 2 sqrt(f) |h(f)| at the measured effective distance
    fH, SH = psd["H1"]
    _, SL = psd["L1"]
    fa = np.geomspace(15, 2000, 400)
    asdH = np.sqrt(np.interp(fa, fH, SH))
    asdL = np.sqrt(np.interp(fa, fH, SL))
    hf = template_fd(M1, M2, F["H1"].f) / summ["Deff_H1_Mpc"]
    hc = 2 * np.sqrt(F["H1"].f) * np.abs(hf)
    hca = np.interp(fa, F["H1"].f, hc)
    # overlay templates on a coarse grid for the page's bank explorer
    sub = []
    for m1 in range(20, 61, 4):
        for m2 in range(15, m1 + 1, 4):
            h = template_fd(m1, m2, F["H1"].f)
            z, sig = F["H1"].snr(h)
            tax = t_axis("H1")
            mm = (tax > -2) & (tax < 2)
            i = int(np.argmax(np.abs(z) * mm))
            amp = abs(z[i]) / sig
            ht = h * amp * np.exp(1j * np.angle(z[i])) * np.exp(-2j * np.pi * F["H1"].f * (i / FS))
            wf = ht / np.sqrt(F["H1"].S * FS / 2) * FS
            wf[(F["H1"].f < 35) | (F["H1"].f > 350)] = 0
            tw = np.fft.irfft(wf, n=F["H1"].N)
            _, tw_w = window(tw, "H1")
            sub.append(dict(m1=m1, m2=m2, snr=round(float(abs(z[i])), 2), w=np.round(tw_w[::2], 3).tolist()))
    # ------------------------------------------------ audio
    def to_audio(sig, shift=0.0, seconds=(-2.0, 0.3), det="H1"):
        tax = t_axis(det)
        m = (tax >= seconds[0]) & (tax <= seconds[1])
        y = sig[m].astype(np.float64)
        y = y * signal.windows.tukey(len(y), 0.1)
        if shift:
            an = signal.hilbert(y)
            y = np.real(an * np.exp(2j * np.pi * shift * np.arange(len(y)) / FS))
        y = signal.resample_poly(y, 375, 32)  # 4096 Hz -> 48000 Hz
        y = y / (np.abs(y).max() + 1e-12) * 0.8
        pad = np.zeros(int(0.4 * 48000))
        return np.concatenate([pad, y, pad])
    os.makedirs(os.path.join(OUT, "audio"), exist_ok=True)
    clips = {"data_H1": w_t("H1"), "template_H1": tmpl["H1"]}
    for name, sig_ in clips.items():
        for shift, tag in ((0.0, "true"), (400.0, "shift400")):
            sf.write(os.path.join(OUT, "audio", f"{name}_{tag}.wav"), to_audio(sig_, shift).astype(np.float32), 48000)
    json.dump(res, open(os.path.join(OUT, "background.json"), "w"), indent=1)
    page = dict(summary={k: v for k, v in summ.items() if k != "bank" and k != "background"},
                bank=summ["bank"], bg_single=summ["background"], slides=res,
                trace=dict(t=np.round(tH, 4).tolist(), H=np.round(wH, 3).tolist(), L=np.round(wL, 3).tolist(),
                           mH=np.round(mH, 3).tolist(), mL=np.round(mL, 3).tolist()),
                raw=dict(t=np.round(tr, 4).tolist(), h=[float(f"{v:.3e}") for v in rawH]),
                snr=dict(t=np.round(ts, 4).tolist(), H=np.round(sH, 2).tolist(), L=np.round(sL, 2).tolist()),
                asd=dict(f=np.round(fa, 2).tolist(), H=[float(f"{v:.3e}") for v in asdH], L=[float(f"{v:.3e}") for v in asdL],
                         hc=[float(f"{v:.3e}") for v in hca]),
                q=dict(t=np.round(qt, 4).tolist(), f=np.round(qf, 2).tolist(),
                       img=base64.b64encode(np.clip(qimg / 25.0 * 255, 0, 255).astype(np.uint8).tobytes()).decode(),
                       shape=list(qimg.shape)),
                overlays=sub)
    json.dump(page, open(os.path.join(OUT, "page_data.json"), "w"), separators=(",", ":"))
    print(f"page data {os.path.getsize(os.path.join(OUT, 'page_data.json')) / 1024:.0f} KB; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
