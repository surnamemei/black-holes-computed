"""End-to-end EHT simulation: truth image -> observation -> dirty image -> RML image.

Run:  .venv/bin/python src/vlbi/simulate.py      (writes out/vlbi/sim.npz and a diagnostics PNG)
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
from src.kerr.thin import render_thin  # noqa: E402
from src.vlbi import eht  # noqa: E402
from src.vlbi.imaging import RMLImager, blur, nxcorr  # noqa: E402

OUT = os.path.join(ROOT, "out", "vlbi")
THETA_G_UAS = 3.82  # GM/(c^2 D) for M = 6.5e9 Msun, D = 16.8 Mpc
FOV_UAS = 160.0
FLUX = 0.6  # Jy


def rotate(img, deg):
    t = torch.tensor(img, dtype=torch.float32)[None, None]
    th = math.radians(deg)
    M = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0]], dtype=torch.float32)[None]
    grid = Fn.affine_grid(M, t.shape, align_corners=False)
    return Fn.grid_sample(t, grid, align_corners=False)[0, 0].numpy()


def truth_image(n_out=256, a=0.94, inc=17.0, gamma=-1.5, sigma=0.5):
    hw = FOV_UAS / 2 / THETA_G_UAS
    tot, per, _ = render_thin(a=a, inc_deg=inc, half_width_M=hw, n_pix=1024, ss=2, gamma=gamma, sigma=sigma, orders=4)
    hi = tot.cpu().numpy().astype(np.float64)
    per = [p.cpu().numpy().astype(np.float64) for p in per]
    # rotate so the Doppler-bright side sits in the south (bottom), as in the 2017 M87* image
    n = hi.shape[0]
    c = (np.arange(n) - (n - 1) / 2)
    X, Y = np.meshgrid(c, -c)
    ang = math.degrees(math.atan2((hi * Y).sum(), (hi * X).sum()))  # direction of the brightness centroid
    rot = -90 - ang
    hi = rotate(hi, -rot)
    per = [rotate(p, -rot) for p in per]
    if (hi * Y).sum() > 0:  # bright side ended up north: rotate by 180 degrees (a rotation, not a mirror)
        hi = hi[::-1, ::-1].copy()
        per = [p[::-1, ::-1].copy() for p in per]
    assert (hi * Y).sum() < 0
    # area-average down to n_out
    f = n // n_out
    lo = hi.reshape(n_out, f, n_out, f).mean(axis=(1, 3))
    lo = lo / lo.sum() * FLUX
    scale = FLUX / hi.sum()
    return lo, hi * scale, [p * scale for p in per]


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    truth, truth_hi, orders = truth_image()
    pix_truth = FOV_UAS / truth.shape[0]
    print(f"truth image {truth.shape}, {pix_truth:.3f} uas/px, flux {truth.sum():.3f} Jy  [{time.time()-t0:.1f}s]")
    # observation: one day, M87 with the stations that can see it (SPT never can: dec +12 deg)
    sched = eht.schedule(jd0=2457854.5, hours=24.0, cadence_min=6.0, el_min=10.0)
    seen = sorted(set(sched["i"]) | set(sched["j"]))
    print("stations observing M87:", [eht.STATIONS[k][0] for k in seen], f"| {len(sched['u'])} visibilities,",
          f"{len(np.unique(sched['scan']))} scans")
    V, Vobs, sig = eht.observe(truth, pix_truth, sched, seed=7)
    snr = np.abs(V) / sig
    print(f"SNR: median {np.median(snr):.1f}, 10th pct {np.percentile(snr, 10):.1f}, max {snr.max():.0f}")
    tri, cp, s_cp = eht.closure_phases(sched, Vobs, sig)
    quads, lca, s_lca = eht.closure_amps(sched, Vobs, sig)
    print(f"{len(cp)} closure phases, {len(lca)} log closure amplitudes")
    # verify closure phases are immune to the station phases
    _, cp_true, _ = eht.closure_phases(sched, V, sig)
    print(f"closure phase (corrupted - true): median |d| {np.degrees(np.median(np.abs(np.angle(np.exp(1j*(cp-cp_true)))))):.2f} deg (noise only)")
    n_img, pix_img = 64, FOV_UAS / 64
    dirty, beam = eht.dirty_image(sched, V, n_img, pix_img)  # calibrated data, for the explanation
    dirty_raw, _ = eht.dirty_image(sched, Vobs, n_img, pix_img)  # what the uncalibrated data would give
    # settings chosen by a small sweep (prior FWHM, regularizer weights): best NXCORR at 20 uas
    imgr = RMLImager(sched, tri, cp, s_cp, quads, lca, s_lca, n=n_img, pix_uas=pix_img, flux=FLUX, prior_fwhm=40.0)
    rounds = [
        (10, 10, 1.0, 3.0, 1e4, 150, 0.0),
        (20, 20, 1.0, 3.0, 1e4, 200, 0.5),
        (30, 30, 1.0, 2.0, 1e4, 300, 0.33),
        (40, 40, 1.0, 2.0, 1e4, 300, 0.25),
    ]
    t1 = time.time()
    rec, snaps, hist = imgr.run(rounds=rounds, snapshot_every=1)
    print(f"RML: {len(snaps)} snapshots in {time.time()-t1:.1f}s; final chi2_cp {hist[-1]['chi_cp']:.2f}, "
          f"chi2_lca {hist[-1]['chi_lca']:.2f}, flux {hist[-1]['flux']:.3f}")
    # compare with the truth at the same pixel scale, blurred to the nominal resolution
    f = truth.shape[0] // n_img
    truth64 = truth.reshape(n_img, f, n_img, f).sum(axis=(1, 3))
    tb = blur(truth64, pix_img, 20.0)
    rb = blur(rec, pix_img, 10.0)
    print(f"NXCORR(reconstruction, truth) both at 20 uas: {nxcorr(blur(rec, pix_img, 20.0), tb):.3f}; "
          f"rec(10uas) vs truth(20uas): {nxcorr(rb, tb):.3f}; dirty vs truth(20uas): {nxcorr(dirty, tb):.3f}")
    np.savez_compressed(os.path.join(OUT, "sim.npz"), truth=truth, truth_hi=truth_hi.astype(np.float32),
                        orders=np.array(orders, dtype=np.float32), truth64=truth64, truth_blur=tb,
                        u=sched["u"], v=sched["v"], t=sched["t"], i=sched["i"], j=sched["j"], scan=sched["scan"],
                        elev=sched["elev"], times=sched["times"], V=V, Vobs=Vobs, sig=sig, dirty=dirty, beam=beam,
                        dirty_raw=dirty_raw, rec=rec, snaps=np.array(snaps, dtype=np.float32), cp=cp, lca=lca,
                        fov=FOV_UAS, theta_g=THETA_G_UAS)
    # diagnostics figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=90)
    ext = [FOV_UAS / 2, -FOV_UAS / 2, -FOV_UAS / 2, FOV_UAS / 2]
    for a_, im, ttl in zip(ax.ravel(), [truth, tb, beam, dirty, dirty_raw, snaps[0], snaps[len(snaps)//3], rec],
                           ["truth", "truth @20uas", "dirty beam", "dirty image", "dirty (uncalibrated)", "RML start", "RML 1/3", "RML final"]):
        a_.imshow(im, cmap="afmhot", extent=ext)
        a_.set_title(ttl)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diagnostics.png"))
    fig2, a2 = plt.subplots(figsize=(6, 6), dpi=90)
    a2.scatter(sched["u"] / 1e9, sched["v"] / 1e9, s=2)
    a2.scatter(-sched["u"] / 1e9, -sched["v"] / 1e9, s=2)
    a2.set_xlabel("u [G$\\lambda$]"); a2.set_ylabel("v [G$\\lambda$]"); a2.invert_xaxis(); a2.set_aspect("equal")
    fig2.savefig(os.path.join(OUT, "uv.png"))
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
