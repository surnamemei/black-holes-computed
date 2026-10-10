"""Allen's chi^2 signal-consistency test and PyCBC-style reweighted SNR, then time slides again.

chi^2: split the template's band into p bins of equal expected SNR^2; for a true signal each bin
contributes z/p, for a glitch the SNR piles up in a few bins.  chi^2 = p sum_k |z_k - z/p|^2 has
2p - 2 degrees of freedom in Gaussian noise.  Reweighted SNR (Babak et al. 2013):
  rho_hat = rho                                if chi2_r <= 1
          = rho / [(1 + chi2_r^3) / 2]^(1/6)   otherwise
Run:  .venv/bin/python chisq.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.ndimage import maximum_filter1d

from analysis import FHI, FLO, FS, GPS0, OUT, T_EVENT, Filter, load, psd_welch, segment, template_fd

P = 16
summ = json.load(open(os.path.join(OUT, "summary.json")))
M1, M2 = summ["best_m1"], summ["best_m2"]


def snr_chisq(F: Filter, h):
    w = np.where(F.band, 1.0 / F.S, 0.0)
    pw = np.abs(h) ** 2 * w
    sigma2 = 4 * pw.sum() * F.df
    cum = np.cumsum(pw) / pw.sum()
    edges = np.searchsorted(cum, np.linspace(0, 1, P + 1)[1:-1])
    bins = np.split(np.arange(len(F.f)), edges)
    zk = []
    for idx in bins:
        integ = np.zeros(F.N, complex)
        integ[idx] = F.sf[idx] * np.conj(h[idx]) * w[idx]
        zk.append(4 * np.fft.ifft(integ) * F.N * F.df / np.sqrt(sigma2))
    zk = np.array(zk)
    z = zk.sum(0)
    chi2 = P * np.sum(np.abs(zk - z / P) ** 2, axis=0)
    return z, chi2 / (2 * P - 2)


def newsnr(rho, chi2r):
    return np.where(chi2r > 1, rho / ((1 + chi2r ** 3) / 2) ** (1 / 6), rho)


def series(x, f, S, win=64.0, keep=48.0):
    N = len(x)
    rho = np.full(N, np.nan)
    rw = np.full(N, np.nan)
    chi = np.full(N, np.nan)
    pad = (win - keep) / 2
    for c in np.arange(GPS0 + win / 2, GPS0 + 4096 - win / 2 + 1e-9, keep):
        seg, t0 = segment(x, c, win)
        if len(seg) < int(win * FS) or not np.all(np.isfinite(seg)):
            continue
        F = Filter(seg, t0, f, S)
        z, c2 = snr_chisq(F, template_fd(M1, M2, F.f))
        i0 = int((t0 - GPS0) * FS)
        a, b = int(pad * FS), int((pad + keep) * FS)
        rho[i0 + a:i0 + b] = np.abs(z[a:b])
        chi[i0 + a:i0 + b] = c2[a:b]
        rw[i0 + a:i0 + b] = newsnr(np.abs(z[a:b]), c2[a:b])
    return rho, chi, rw


def slides(H, L, tt, step=3, n=1500):
    w = int(0.010 * FS)
    LmaxW = maximum_filter1d(np.nan_to_num(L, nan=0.0), size=2 * w + 1, mode="nearest")
    validL = maximum_filter1d(np.isfinite(L).astype(np.uint8), size=2 * w + 1) > 0
    Hn = np.nan_to_num(H, nan=0.0)
    near = np.abs(tt - T_EVENT) < 1.0
    out, live = [], 0.0
    for s in [s for s in range(-n, n + 1, step) if s != 0]:
        Ls = np.roll(LmaxW, -s * FS)
        # roll the L1 event mask exactly like the L1 data (np.roll wraps around the file end)
        ok = np.isfinite(H) & np.roll(validL, -s * FS) & ~near & ~np.roll(near, -s * FS)
        net = np.where(ok, np.sqrt(Hn ** 2 + Ls ** 2), 0.0)
        out.append(float(net.max()))
        live += ok.sum() / FS
    return np.array(out), live


def main():
    t0 = time.time()
    res = {}
    S = {}
    for det in ("H1", "L1"):
        x = load(det)
        f, Sd, _ = psd_welch(x)
        S[det] = series(x, f, Sd)
        print(f"{det}: SNR, chi^2 and reweighted SNR series done ({time.time() - t0:.0f}s)")
    tt = GPS0 + np.arange(len(S["H1"][0])) / FS
    ev = np.abs(tt - T_EVENT) < 0.1
    for det in ("H1", "L1"):
        rho, chi, rw = S[det]
        i = int(np.nanargmax(np.where(ev, rho, -1)))
        res[f"event_{det}"] = dict(snr=float(rho[i]), chi2r=float(chi[i]), newsnr=float(rw[i]))
        off = np.where(np.abs(tt - T_EVENT) > 2, rho, -1)
        j = int(np.nanargmax(off))
        res[f"loudest_offsource_{det}"] = dict(t=float(tt[j] - T_EVENT), snr=float(rho[j]), chi2r=float(chi[j]), newsnr=float(rw[j]))
        print(f"{det}: event SNR {rho[i]:.2f}, chi2_r {chi[i]:.2f}, reweighted {rw[i]:.2f} | loudest off-source "
              f"SNR {rho[j]:.2f} at {tt[j] - T_EVENT:+.1f} s, chi2_r {chi[j]:.1f}, reweighted {rw[j]:.2f}")
    res["event_network_newsnr"] = float(np.hypot(res["event_H1"]["newsnr"], res["event_L1"]["newsnr"]))
    raw, live = slides(S["H1"][0], S["L1"][0], tt)
    rew, _ = slides(S["H1"][2], S["L1"][2], tt)
    res.update(slides=len(raw), live_days=live / 86400, loudest_raw=raw.round(3).tolist(), loudest_reweighted=rew.round(3).tolist(),
               max_raw=float(raw.max()), max_reweighted=float(rew.max()))
    print(f"time slides ({len(raw)}, {live / 86400:.1f} days): loudest background network SNR {raw.max():.2f} raw, "
          f"{rew.max():.2f} reweighted; event reweighted network SNR {res['event_network_newsnr']:.2f}")
    json.dump(res, open(os.path.join(OUT, "chisq.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
