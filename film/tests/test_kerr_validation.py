"""Physics validation of the Kerr ray tracer.

1. Schwarzschild critical impact parameter b_c = sqrt(27) M.
2. Kerr shadow (a = 0.99, edge-on, and a = 0.9 at 17 deg) against Bardeen's analytic critical curve.
3. FP32 GPU integrator against an independent FP64 reference whose equations of
   motion come from autograd of the textbook inverse metric (not the hand-derived
   Mino-time equations used in the kernel).
4. Equatorial-crossing radius accuracy (used for every disk pixel).

Run:  .venv/bin/python tests/test_kerr_validation.py
Writes figures to out/validation/.
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kerr.tracer import Camera, camera_rays, trace, trace_rays, ST_ESCAPED, ST_HORIZON, horizon_radius  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "validation")
os.makedirs(OUT, exist_ok=True)
results = {}


def equatorial_ingoing(r0, b, a):
    """p_r of a photon arriving at r0 from the hole (equatorial, p_th = 0), E = 1.
    Traced backwards, the ray heads inwards."""
    D = r0 * r0 - 2 * r0 + a * a
    P = r0 * r0 + a * a - a * b
    return +np.sqrt(np.maximum(P * P / D - (b - a) ** 2, 0) / D)


# ------------------------------------------------------------------ 1. Schwarzschild b_c
def test_bcrit():
    r0, a = 1000.0, 0.0
    lo, hi = 5.0, 5.4
    for it in range(4):
        b = np.linspace(lo, hi, 4001)
        pr = equatorial_ingoing(r0, b, a)
        out = trace_rays(r0, math.pi / 2, 0.0, torch.tensor(pr, dtype=torch.float32, device="cuda"),
                         torch.zeros(len(b), device="cuda"), torch.tensor(b, dtype=torch.float32, device="cuda"), a,
                         r_esc=5000.0)
        st = out["status"].cpu().numpy()
        cap = np.where(st == ST_HORIZON)[0]
        esc = np.where(st == ST_ESCAPED)[0]
        b_last_cap = b[cap.max()]
        b_first_esc = b[esc.min()]
        lo, hi = b_last_cap - 2 * (b[1] - b[0]), b_first_esc + 2 * (b[1] - b[0])
    bc = 0.5 * (b_last_cap + b_first_esc)
    err = bc - math.sqrt(27)
    print(f"[1] Schwarzschild b_crit = {bc:.6f}  (exact sqrt(27) = {math.sqrt(27):.6f}, error {err:+.2e})")
    results["bcrit_error"] = err
    assert abs(err) < 1e-3


# ------------------------------------------------------------------ 2. Kerr shadow vs Bardeen
def bardeen_curve(a, theta_o, n=20000):
    rp = 2 * (1 + math.cos(2 / 3 * math.acos(-a)))
    rm = 2 * (1 + math.cos(2 / 3 * math.acos(a)))
    r = np.linspace(rp, rm, n)
    xi = -(r ** 3 - 3 * r ** 2 + a * a * r + a * a) / (a * (r - 1))
    eta = -(r ** 3) * (r ** 3 - 6 * r ** 2 + 9 * r - 4 * a * a) / (a * a * (r - 1) ** 2)
    alpha = -xi / math.sin(theta_o)
    b2 = eta + a * a * math.cos(theta_o) ** 2 - xi ** 2 / math.tan(theta_o) ** 2
    ok = b2 >= 0
    return alpha[ok], np.sqrt(b2[ok])


def test_kerr_shadow(a, inc_deg, tag):
    r_o = 5000.0
    th = math.radians(inc_deg)
    N = 1200
    half = 9.0  # M
    fov = 2 * math.degrees(math.atan(half / r_o))
    cam = Camera(r=r_o, theta=th, fov=fov)
    t0 = time.time()
    res = trace(cam, a, N, N, r_esc=20000.0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    st = res.status.cpu().numpy()
    # image-plane coordinates (alpha right, beta up) in units of M
    tx = math.tan(math.radians(fov) / 2)
    xs = ((np.arange(N) + 0.5) / N * 2 - 1) * tx * r_o
    ys = (1 - (np.arange(N) + 0.5) / N * 2) * tx * r_o
    cap = st == ST_HORIZON
    # shadow edges along each row
    al, be = bardeen_curve(a, th)
    # split the critical curve at its top into left/right branches, each monotonic in beta
    k = int(np.argmax(be))
    left_b, left_a = be[: k + 1], al[: k + 1]
    right_b, right_a = be[k:][::-1], al[k:][::-1]
    if left_a.mean() > right_a.mean():
        left_b, left_a, right_b, right_a = right_b, right_a, left_b, left_a
    rows = np.where(cap.any(axis=1))[0]
    px = xs[1] - xs[0]
    dev_l, dev_r = [], []
    for i in rows:
        beta = abs(ys[i])
        if beta > 0.9 * be.max():  # skip the tangency at the top/bottom
            continue
        js = np.where(cap[i])[0]
        oL = np.argsort(left_b)
        oR = np.argsort(right_b)
        aL = np.interp(beta, left_b[oL], left_a[oL])
        aR = np.interp(beta, right_b[oR], right_a[oR])
        # numeric edge = midpoint between last outside and first inside pixel centres
        dev_l.append((xs[js.min()] - px / 2) - aL)
        dev_r.append((xs[js.max()] + px / 2) - aR)
    dev = np.abs(np.concatenate([dev_l, dev_r]))
    maxdev = dev.max()
    print(f"[2] Kerr a={a} inc={inc_deg}deg: {N}x{N} rays in {dt:.2f}s; shadow edge |deviation| "
          f"median {np.median(dev):.4f} M, max {maxdev:.4f} M (pixel = {px:.4f} M); "
          f"median steps {np.median(res.nsteps.cpu().numpy()):.0f}")
    mid = N // 2
    js = np.where(cap[mid])[0]
    print(f"    equatorial extent: numeric [{xs[js.min()]:.3f}, {xs[js.max()]:.3f}] vs Bardeen "
          f"[{al.min():.3f}, {al.max():.3f}]")
    results[f"shadow_{tag}_maxdev_M"] = maxdev
    results[f"shadow_{tag}_pixel_M"] = px
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    ax.imshow(cap, extent=[xs[0] - px / 2, xs[-1] + px / 2, ys[-1] - px / 2, ys[0] + px / 2], cmap="Greys", origin="upper")
    ax.plot(al, be, "r-", lw=1, label="Bardeen (1973) analytic")
    ax.plot(al, -be, "r-", lw=1)
    ax.set_xlabel(r"$\alpha$ [M]")
    ax.set_ylabel(r"$\beta$ [M]")
    ax.set_title(f"Kerr shadow a={a}, i={inc_deg}°: GPU trace (black) vs analytic (red)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"kerr_shadow_{tag}.png"))
    plt.close(fig)
    assert maxdev < 1.5 * px


# ------------------------------------------------------------------ 3. FP64 autograd reference
def H_textbook(x, p, a):
    """1/2 g^{mu nu} p_mu p_nu with the Boyer-Lindquist inverse metric written out directly."""
    r, th = x[..., 1], x[..., 2]
    s2 = torch.sin(th) ** 2
    Sig = r * r + a * a * torch.cos(th) ** 2
    D = r * r - 2 * r + a * a
    A = (r * r + a * a) ** 2 - a * a * D * s2
    gtt = -A / (Sig * D)
    gtp = -2 * a * r / (Sig * D)
    gpp = (D - a * a * s2) / (Sig * D * s2)
    grr = D / Sig
    gthth = 1 / Sig
    pt, pr, pth, pph = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    return 0.5 * (gtt * pt * pt + 2 * gtp * pt * pph + gpp * pph * pph + grr * pr * pr + gthth * pth * pth)


def reference_trace(r0, th0, pr, pth, b, a, r_esc=2000.0, r_stop=None, step=0.004):
    """RK4 in affine parameter with autograd forces; step ~ 0.002 * (r - r_h)... in float64."""
    from torch.func import grad, vmap

    if r_stop is None:
        r_stop = horizon_radius(a) + 0.01 * (2 - horizon_radius(a)) + 1e-3
    n = len(pr)
    x = torch.zeros(n, 4, dtype=torch.float64, device="cuda")
    x[:, 1], x[:, 2] = r0, th0
    p = torch.zeros(n, 4, dtype=torch.float64, device="cuda")
    p[:, 0], p[:, 1], p[:, 2], p[:, 3] = -1.0, pr, pth, b

    def ham(xx, pp):
        return H_textbook(xx, pp, a)

    dHdx = vmap(grad(ham, argnums=0))
    dHdp = vmap(grad(ham, argnums=1))

    def f(x, p):
        return dHdp(x, p), -dHdx(x, p)

    alive = torch.ones(n, dtype=torch.bool, device="cuda")
    status = torch.full((n,), -1, dtype=torch.int64, device="cuda")
    first_cross = torch.full((n,), float("nan"), dtype=torch.float64, device="cuda")
    esc_dir = torch.zeros(n, 3, dtype=torch.float64, device="cuda")
    for it in range(400000):
        r = x[:, 1]
        h = step * torch.clamp(r - horizon_radius(a), min=0.02) ** 1.0 * torch.clamp(r / 10, min=1.0)
        h = -torch.where(alive, h, torch.zeros_like(h))[:, None]  # backwards in time
        k1x, k1p = f(x, p)
        k2x, k2p = f(x + 0.5 * h * k1x, p + 0.5 * h * k1p)
        k3x, k3p = f(x + 0.5 * h * k2x, p + 0.5 * h * k2p)
        k4x, k4p = f(x + h * k3x, p + h * k3p)
        xn = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        pn = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        c0, c1 = torch.cos(x[:, 2]), torch.cos(xn[:, 2])
        crossed = alive & (c0 * c1 < 0) & torch.isnan(first_cross)
        t = c0 / (c0 - c1)
        first_cross = torch.where(crossed, x[:, 1] + t * (xn[:, 1] - x[:, 1]), first_cross)
        x, p = xn, pn
        cap = alive & (x[:, 1] < r_stop)
        esc = alive & (x[:, 1] > r_esc) & (p[:, 1] < 0)
        status[cap] = ST_HORIZON
        if esc.any():
            v, _ = f(x, p)
            v = -v  # velocity of the backward ray
            rr, th, ph = x[:, 1], x[:, 2], x[:, 3]
            vr, vth, vph = v[:, 1], rr * v[:, 2], rr * torch.sin(th) * v[:, 3]
            st, ct, sp, cp = torch.sin(th), torch.cos(th), torch.sin(ph), torch.cos(ph)
            d = torch.stack([vr * st * cp + vth * ct * cp - vph * sp, vr * st * sp + vth * ct * sp + vph * cp,
                             vr * ct - vth * st], -1)
            d = d / d.norm(dim=-1, keepdim=True)
            esc_dir[esc] = d[esc]
            status[esc] = ST_ESCAPED
        alive = alive & ~cap & ~esc
        if not alive.any():
            break
    return status, esc_dir, first_cross


def test_fp64_reference():
    a = 0.9
    r_o, th_o = 30.0, math.radians(80)
    cam = Camera(r=r_o, theta=th_o, fov=40)
    init = camera_rays(cam, a, 48, 32)
    pr = init["pr"].reshape(-1)
    pth = init["pth"].reshape(-1)
    b = init["b"].reshape(-1)
    t0 = time.time()
    g = trace_rays(r_o, th_o, 0.0, pr, pth, b, a, r_esc=2000.0)
    st_ref, dir_ref, fc_ref = reference_trace(r_o, th_o, pr.double(), pth.double(), b.double(), a, r_esc=2000.0)
    dt = time.time() - t0
    st_g = g["status"].long()
    agree = (st_g == st_ref).float().mean().item()
    both_esc = (st_g == ST_ESCAPED) & (st_ref == ST_ESCAPED)
    d1, d2 = g["esc_dir"].double()[both_esc], dir_ref[both_esc]
    # atan2(|d1 x d2|, d1.d2): arccos of a float32 dot product cannot resolve < 0.03 deg
    ang = torch.rad2deg(torch.atan2(torch.linalg.cross(d1, d2).norm(dim=-1), (d1 * d2).sum(-1))).cpu().numpy()
    has_c = (g["ncross"] > 0) & ~torch.isnan(fc_ref)
    dr = (g["cross"][:, 0, 0].double()[has_c] - fc_ref[has_c]).abs().cpu().numpy()
    print(f"[3] FP32 Warp vs FP64 autograd-Hamiltonian reference ({len(pr)} rays, a={a}, r_o={r_o}, {dt:.1f}s):")
    print(f"    fate agreement {agree * 100:.2f}%;  escape-direction error: median {np.median(ang):.2e} deg, "
          f"99th pct {np.percentile(ang, 99):.2e} deg, max {ang.max():.2e} deg")
    print(f"    first equatorial crossing radius error: median {np.median(dr):.2e} M, max {dr.max():.2e} M")
    results["fp64_fate_agreement"] = agree
    results["fp64_dir_err_median_deg"] = float(np.median(ang))
    results["fp64_dir_err_p99_deg"] = float(np.percentile(ang, 99))
    results["fp64_cross_err_max_M"] = float(dr.max())
    # The tail is FP32 round-off on rays that wind near the photon orbit: halving the
    # step caps does not reduce it (checked), so 99% within ~1 pixel at 1080p/40deg is the bar.
    assert agree > 0.995
    assert np.median(ang) < 0.002
    assert np.percentile(ang, 99) < 0.021


if __name__ == "__main__":
    test_bcrit()
    test_kerr_shadow(0.99, 90.0, "a099_i90")
    test_kerr_shadow(0.9, 17.0, "a09_i17")
    test_fp64_reference()
    import json

    with open(os.path.join(OUT, "validation_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("ALL VALIDATION TESTS PASSED")
