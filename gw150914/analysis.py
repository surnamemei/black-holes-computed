"""Detect GW150914 in LIGO's public O1 strain with a from-scratch matched-filter pipeline.

Steps (all numpy/scipy except the waveform model):
  1. noise PSD per detector: median-averaged Welch over the 4096 s file, excluding +-32 s of the event
  2. frequency-domain matched filter against an IMRPhenomD template bank (LALSimulation),
     complex SNR z(t) = 4 sum s(f) h*(f) / S(f) e^{2 pi i f t} df / sigma_h, 30-1024 Hz
  3. network coincidence: rho_net = sqrt(rho_H^2 + rho_L^2) with |t_H - t_L| <= 10 ms
  4. background: the best template slid over the rest of the data (off-source noise peaks)
  5. products: whitened traces, spectrogram, SNR series, bank map, audio, JSON summary

Run:  .venv/bin/python analysis.py
"""
from __future__ import annotations

import json
import math
import os
import time

import h5py
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
FS = 4096
T_EVENT = 1126259462.4
GPS0 = 1126256640
FLO, FHI = 30.0, 1024.0
SEG = 32.0  # analysis segment (s)
MSUN_S = 4.925490947e-6  # G Msun / c^3 in seconds


def load(det):
    f = os.path.join(HERE, "data", f"{det[0]}-{det}_LOSC_4_V1-1126256640-4096.hdf5")
    with h5py.File(f, "r") as h:
        x = h["strain/Strain"][:].astype(np.float64)
    return x


def psd_welch(x, seg_s=8.0, exclude=(T_EVENT - 32, T_EVENT + 32)):
    """Median-averaged Welch PSD (one-sided) from all finite segments outside the exclusion window."""
    n = int(seg_s * FS)
    hop = n // 2
    win = signal.windows.hann(n)
    U = (win ** 2).sum()
    specs = []
    for i0 in range(0, len(x) - n, hop):
        t0 = GPS0 + i0 / FS
        if t0 + seg_s > exclude[0] and t0 < exclude[1]:
            continue
        seg = x[i0:i0 + n]
        if not np.all(np.isfinite(seg)):
            continue
        X = np.fft.rfft((seg - seg.mean()) * win)
        specs.append(2 * np.abs(X) ** 2 / (FS * U))
    S = np.median(np.array(specs), axis=0) / np.log(2.0)  # median -> mean bias for chi^2_2
    f = np.fft.rfftfreq(n, 1 / FS)
    return f, S, len(specs)


def segment(x, t_center, dur=SEG):
    i0 = int((t_center - dur / 2 - GPS0) * FS)
    return x[i0:i0 + int(dur * FS)], GPS0 + i0 / FS


def template_fd(m1, m2, f, chi1=0.0, chi2=0.0):
    """IMRPhenomD frequency-domain waveform h+(f) at 1 Mpc, face-on, on frequency grid f (Hz)."""
    import lal
    import lalsimulation as ls
    df = f[1] - f[0]
    params = lal.CreateDict()
    hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, chi1, 0, 0, chi2,
                                            1e6 * lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0, df, 20.0, f[-1], 20.0,
                                            params, ls.IMRPhenomD)
    h = np.zeros(len(f), complex)
    n = min(len(f), hp.data.length)
    h[:n] = hp.data.data[:n]
    return h


class Filter:
    """Matched filter on one detector's analysis segment."""

    def __init__(self, x_seg, t0, f_psd, S_psd):
        self.N = len(x_seg)
        self.t0 = t0
        win = signal.windows.tukey(self.N, alpha=1 / 8)
        self.f = np.fft.rfftfreq(self.N, 1 / FS)
        self.df = self.f[1] - self.f[0]
        self.S = np.interp(self.f, f_psd, S_psd)
        self.sf = np.fft.rfft(x_seg * win) / FS  # continuous-FT normalisation
        self.band = (self.f >= FLO) & (self.f <= FHI)

    def snr(self, h):
        """Complex SNR time series for template h(f) (defined on self.f), normalised to unit noise."""
        w = np.where(self.band, 1.0 / self.S, 0.0)
        sigma2 = 4 * np.sum(np.abs(h) ** 2 * w) * self.df
        integrand = np.zeros(self.N, complex)
        integrand[:len(self.f)] = self.sf * np.conj(h) * w  # positive frequencies only -> analytic signal
        z = 4 * np.fft.ifft(integrand) * self.N * self.df / np.sqrt(sigma2)
        return z, math.sqrt(sigma2)

    def whitened(self, flo=35.0, fhi=350.0):
        """Whitened, band-passed time series of the segment (unit-variance in band)."""
        wf = self.sf / np.sqrt(self.S * FS / 2) * FS
        wf[(self.f < flo) | (self.f > fhi)] = 0
        return np.fft.irfft(wf, n=self.N)


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def main():
    t_start = time.time()
    res = {}
    data, psd, flt = {}, {}, {}
    for det in ("H1", "L1"):
        x = load(det)
        f, S, nseg = psd_welch(x)
        psd[det] = (f, S)
        seg, t0 = segment(x, T_EVENT)
        assert np.all(np.isfinite(seg)), f"{det}: NaNs in the analysis segment"
        data[det] = x
        flt[det] = Filter(seg, t0, f, S)
        res[f"{det}_psd_segments"] = nseg
        print(f"{det}: PSD from {nseg} x 8 s segments; ASD at 100 Hz = {math.sqrt(np.interp(100, f, S)):.2e} /rtHz")
    # ---------------------------------------------------------------- template bank
    fgrid = flt["H1"].f
    bank = []
    m1s = np.arange(20.0, 61.0, 1.0)
    for m1 in m1s:
        for m2 in np.arange(15.0, m1 + 0.01, 1.0):
            bank.append((m1, m2))
    print(f"bank: {len(bank)} IMRPhenomD templates (detector-frame masses, non-spinning)")
    win_ms = 0.010
    best = None
    rows = []
    tb = time.time()
    t_lo, t_hi = 4.0, SEG - 4.0  # ignore edges (wrap-around and taper)
    for k, (m1, m2) in enumerate(bank):
        h = template_fd(m1, m2, fgrid)
        out = {}
        for det in ("H1", "L1"):
            z, sig = flt[det].snr(h)
            t = np.arange(flt[det].N) / FS
            m = (t > t_lo) & (t < t_hi)
            out[det] = (np.abs(z) * m, z)
        # network: for each H1 time, best L1 within +-10 ms
        aH = out["H1"][0]
        aL = out["L1"][0]
        w = int(win_ms * FS)
        aLmax = maximum_filter1d(aL, size=2 * w + 1, mode="wrap")  # best L1 SNR within +-10 ms
        net = np.sqrt(aH ** 2 + aLmax ** 2)
        i = int(np.argmax(net))
        rows.append((m1, m2, float(net[i]), float(aH[i]), float(aLmax[i])))
        if best is None or net[i] > best["net"]:
            best = dict(m1=m1, m2=m2, net=float(net[i]), iH=i, H=float(aH[i]))
        if k % 200 == 0:
            print(f"  {k}/{len(bank)} templates, {time.time() - tb:.0f}s; best so far {best['m1']:.0f}+{best['m2']:.0f} Msun, net SNR {best['net']:.2f}")
    # refine the best template's L1 peak and the time / phase offsets
    h = template_fd(best["m1"], best["m2"], fgrid)
    zH, sH = flt["H1"].snr(h)
    zL, sL = flt["L1"].snr(h)
    w = int(win_ms * FS)
    iH = best["iH"]
    cand = [(abs(zL[iH + s]), iH + s) for s in range(-w, w + 1)]
    aLb, iL = max(cand)
    tH = flt["H1"].t0 + iH / FS
    tL = flt["L1"].t0 + iL / FS
    res.update(best_m1=best["m1"], best_m2=best["m2"], chirp_mass=chirp_mass(best["m1"], best["m2"]),
               snr_H1=float(abs(zH[iH])), snr_L1=float(aLb), snr_network=float(math.hypot(abs(zH[iH]), aLb)),
               t_H1=tH, t_L1=tL, dt_ms=(tH - tL) * 1e3, dphase_deg=float(np.degrees(np.angle(zH[iH] / zL[iL]))),
               sigma_H1_Mpc=sH, sigma_L1_Mpc=sL)
    # effective distance: template at 1 Mpc gives SNR = sigma; observed SNR rho -> D_eff = sigma / rho Mpc
    res["Deff_H1_Mpc"] = sH / res["snr_H1"]
    res["Deff_L1_Mpc"] = sL / res["snr_L1"]
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    # ---------------------------------------------------------------- chirp-mass likelihood from the bank
    R = np.array(rows)
    res["bank"] = [[float(a), float(b), round(float(c), 3)] for a, b, c, _, _ in rows]
    rho2 = R[:, 2] ** 2
    L = np.exp(0.5 * (rho2 - rho2.max()))
    mc = chirp_mass(R[:, 0], R[:, 1])
    order = np.argsort(mc)
    cdf = np.cumsum(L[order]) / L.sum()
    res["chirp_mass_median"] = float(np.interp(0.5, cdf, mc[order]))
    res["chirp_mass_90"] = [float(np.interp(0.05, cdf, mc[order])), float(np.interp(0.95, cdf, mc[order]))]
    print(f"chirp-mass (detector frame) from exp(rho^2/2) over the bank: {res['chirp_mass_median']:.1f} "
          f"[{res['chirp_mass_90'][0]:.1f}, {res['chirp_mass_90'][1]:.1f}] Msun (90%)")
    # ---------------------------------------------------------------- background: best template off-source
    bg = {}
    for det in ("H1", "L1"):
        x = data[det]
        f, S = psd[det]
        peaks = []
        for c in np.arange(GPS0 + 40, GPS0 + 4096 - 40, 24.0):
            if abs(c - T_EVENT) < 40:
                continue
            seg, t0 = segment(x, c)
            if not np.all(np.isfinite(seg)):
                continue
            F = Filter(seg, t0, f, S)
            z, _ = F.snr(template_fd(best["m1"], best["m2"], F.f))
            tt = np.arange(F.N) / FS
            mm = (tt > 4) & (tt < 28)
            peaks.append(float(np.abs(z[mm]).max()))
        bg[det] = peaks
        print(f"{det} background: {len(peaks)} off-source 24 s windows; loudest noise SNR {max(peaks):.2f}, median {np.median(peaks):.2f}")
    res["background"] = bg
    # ---------------------------------------------------------------- products for plots / audio / page
    np.savez_compressed(os.path.join(OUT, "products.npz"),
                        fH=psd["H1"][0], SH=psd["H1"][1], fL=psd["L1"][0], SL=psd["L1"][1],
                        wH=flt["H1"].whitened(), wL=flt["L1"].whitened(), t0H=flt["H1"].t0, t0L=flt["L1"].t0,
                        zH=zH, zL=zL, bank=R)
    # best-fit template, whitened and aligned, for overlays
    for det, z, i, F in (("H1", zH, iH, flt["H1"]), ("L1", zL, iL, flt["L1"])):
        amp = abs(z[i]) / (F.snr(h)[1])  # amplitude scale: rho / sigma
        phase = np.angle(z[i])
        ht = h * amp * np.exp(1j * phase) * np.exp(-2j * np.pi * F.f * (i / FS))
        wf = ht / np.sqrt(F.S * FS / 2) * FS
        wf[(F.f < 35) | (F.f > 350)] = 0
        res[f"_tmpl_{det}"] = np.fft.irfft(wf, n=F.N)
    np.savez_compressed(os.path.join(OUT, "templates.npz"), H1=res.pop("_tmpl_H1"), L1=res.pop("_tmpl_L1"))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
