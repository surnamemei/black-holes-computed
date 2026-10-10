"""Optically thin equatorial emission (an M87*-like 1.3 mm image model).

Every crossing of the equatorial plane adds g^3 J(r) (flat spectrum, I_nu / nu^3
invariant), so the image shows the direct image (n = 0) plus the nested photon-ring
images (n = 1, 2, ...).  The emissivity profile J(r) is the Johnson-SU family used by
Gralla, Lupsasca & Marrone (2020, PRD 102, 124004); emitters follow Keplerian orbits
outside the ISCO and Cunningham plunging geodesics inside it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from .emission import Disk
from .tracer import MAX_CROSS, Camera, horizon_radius, trace


def johnson_su(r, mu, gamma, sigma):
    x = (r - mu) / sigma
    return torch.exp(-0.5 * (gamma + torch.asinh(x)) ** 2) / torch.sqrt((r - mu) ** 2 + sigma ** 2)


def render_thin(a=0.94, inc_deg=17.0, half_width_M=20.0, n_pix=1024, ss=3, r_obs=5000.0,
                mu=None, gamma=-1.5, sigma=0.5, orders=3, roll_deg=0.0):
    """Returns (total image, list of per-order images), each (n_pix, n_pix), plus extent in M.

    The camera is at inclination inc_deg; image x points along +alpha (screen right),
    y along +beta (screen up).  roll_deg rotates the camera about the line of sight.
    """
    fov = 2 * math.degrees(math.atan(half_width_M / r_obs))
    th = math.radians(inc_deg)
    rr = math.radians(roll_deg)
    up = (0.0, -math.cos(rr), math.sin(rr))
    cam = Camera(r=r_obs, theta=th, fov=fov, up=up)
    res = trace(cam, a, n_pix, n_pix, ss=ss, r_esc=4 * r_obs)
    disk = Disk(a)
    if mu is None:
        mu = horizon_radius(a)
    H, W = res.status.shape
    N = H * W
    cross = res.cross.view(N, MAX_CROSS, 4)
    nc = res.ncross.view(-1)
    b = res.b.view(-1)
    E = res.E_cam.view(-1)
    per = []
    for k in range(orders):
        m = nc > k
        img = torch.zeros(N, device="cuda")
        if m.any():
            idx = m.nonzero().squeeze(1)
            r = cross[idx, k, 0]
            g = disk.redshift(r, b[idx], cross[idx, k, 2], E[idx])
            J = johnson_su(r, mu, gamma, sigma)
            img[idx] = g ** 3 * J * (r > horizon_radius(a)).float()
        per.append(img.view(H, W))
    per = [Fn.avg_pool2d(p[None, None], ss)[0, 0] for p in per]
    total = sum(per)
    return total, per, half_width_M


AFMHOT = None


def afmhot(x):
    """matplotlib's 'afmhot' colormap (x in [0, 1]) -> (..., 3)."""
    x = x.clamp(0, 1)
    r = (2 * x).clamp(0, 1)
    g = (2 * x - 0.5).clamp(0, 1)
    b = (2 * x - 1.0).clamp(0, 1)
    return torch.stack([r, g, b], -1)
