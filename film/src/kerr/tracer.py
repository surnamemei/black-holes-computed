"""Null-geodesic tracer for the Kerr metric, running on the GPU via NVIDIA Warp.

Units: G = c = M = 1. Boyer-Lindquist coordinates (t, r, theta, phi), spin 0 <= a < 1.

Photons are integrated backwards from the camera (time reversal is free for null
geodesics: we simply launch the ray along the direction the camera looks).  With
E = -p_t = 1 and b = p_phi / E, the super-Hamiltonian multiplied by rho^2 is

    H~ = 1/2 [ Delta p_r^2 + p_th^2 + W(r, th) ],
    W  = -P^2/Delta + b^2/sin^2(th) - 2ab + a^2 sin^2(th),   P = r^2 + a^2 - ab,

and Hamilton's equations in Mino time (d tau = rho^2 d lambda) are exact on the
constraint surface H~ = 0:

    dr/dl    = Delta p_r
    dth/dl   = p_th
    dphi/dl  = a P / Delta + b / sin^2(th) - a
    dp_r/dl  = -(r-1) p_r^2 + 2 r P / Delta - (r-1) P^2 / Delta^2
    dp_th/dl = b^2 cos(th) / sin^3(th) - a^2 sin(th) cos(th)

The state always holds the photon's true (future-directed) covariant momentum;
the integration runs backwards in Mino time (the RHS is negated), which in Kerr
is NOT equivalent to flipping the spatial momentum (that would flip the spin).

Integration uses adaptive Dormand-Prince 5(4) with per-step caps on the change
of every coordinate. Every crossing of the equatorial plane (cos th = 0) is
located by cubic Hermite interpolation and recorded, so one trace can be
re-shaded with any equatorial emission model (opaque Novikov-Thorne disk, or
optically thin emission that reveals the n = 0, 1, 2... photon-ring images).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

wp.config.quiet = True
wp.init()

vec5 = wp.types.vector(length=5, dtype=wp.float32)

MAX_CROSS = 6  # equatorial crossings recorded per ray

# status codes
ST_ESCAPED = 0
ST_HORIZON = 1
ST_DISK = 2  # terminated on an opaque disk
ST_MAXSTEP = 3


def horizon_radius(a: float) -> float:
    return 1.0 + math.sqrt(1.0 - a * a)


def isco_radius(a: float, prograde: bool = True) -> float:
    """Bardeen-Press-Teukolsky ISCO radius."""
    z1 = 1.0 + (1.0 - a * a) ** (1 / 3) * ((1.0 + a) ** (1 / 3) + (1.0 - a) ** (1 / 3))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    s = -1.0 if prograde else 1.0
    return 3.0 + z2 + s * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


@wp.func
def rhs(y: vec5, a: float, b: float) -> vec5:
    r = y[0]
    th = y[1]
    pr = y[3]
    pth = y[4]
    s = wp.sin(th)
    c = wp.cos(th)
    if wp.abs(s) < 1.0e-6:
        if s < 0.0:
            s = -1.0e-6
        else:
            s = 1.0e-6
    s2 = s * s
    D = r * r - 2.0 * r + a * a
    P = r * r + a * a - a * b
    dr = D * pr
    dth = pth
    dph = a * P / D + b / s2 - a
    dpr = -(r - 1.0) * pr * pr + 2.0 * r * P / D - (r - 1.0) * P * P / (D * D)
    dpth = b * b * c / (s2 * s) - a * a * s * c
    # rays are traced backwards in time from the camera: d/d(-lambda)
    return vec5(-dr, -dth, -dph, -dpr, -dpth)


@wp.func
def hermite(y0: float, y1: float, d0: float, d1: float, h: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * h * d0 + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * h * d1


@wp.func
def hermite_d(y0: float, y1: float, d0: float, d1: float, h: float, t: float) -> float:
    # derivative of hermite() with respect to t
    t2 = t * t
    return (6.0 * t2 - 6.0 * t) * y0 + (3.0 * t2 - 4.0 * t + 1.0) * h * d0 + (-6.0 * t2 + 6.0 * t) * y1 + (3.0 * t2 - 2.0 * t) * h * d1


@wp.kernel
def trace_kernel(
    pr0: wp.array(dtype=wp.float32),
    pth0: wp.array(dtype=wp.float32),
    b_arr: wp.array(dtype=wp.float32),
    r0: float,
    th0: float,
    ph0: float,
    a: float,
    r_stop: float,
    r_esc: float,
    disk_rin: float,
    disk_rout: float,
    max_steps: int,
    rtol: float,
    atol: float,
    cap_scale: float,
    status: wp.array(dtype=wp.int32),
    esc_dir: wp.array(dtype=wp.vec3),
    cross: wp.array2d(dtype=wp.vec4),
    ncross: wp.array(dtype=wp.int32),
    nsteps: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    b = b_arr[tid]
    y = vec5(r0, th0, ph0, pr0[tid], pth0[tid])
    k1 = vec5(rhs(y, a, b))
    h = float(0.02 / r0)
    st = int(-1)
    nc = int(0)
    i = int(0)
    while st < 0 and i < max_steps:
        i += 1
        # --- step-size caps: limit the change of each coordinate per step
        r = y[0]
        cap = cap_scale * 0.05 * r / (wp.abs(k1[0]) + 1.0e-12)
        cap = wp.min(cap, cap_scale * 0.04 / (wp.abs(k1[1]) + 1.0e-12))
        cap = wp.min(cap, cap_scale * 0.08 / (wp.abs(k1[2]) + 1.0e-12))
        h = wp.min(h, cap)
        # --- Dormand-Prince 5(4)
        k2 = rhs(y + h * (0.2 * k1), a, b)
        k3 = rhs(y + h * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2), a, b)
        k4 = rhs(y + h * ((44.0 / 45.0) * k1 - (56.0 / 15.0) * k2 + (32.0 / 9.0) * k3), a, b)
        k5 = rhs(y + h * ((19372.0 / 6561.0) * k1 - (25360.0 / 2187.0) * k2 + (64448.0 / 6561.0) * k3 - (212.0 / 729.0) * k4), a, b)
        k6 = rhs(y + h * ((9017.0 / 3168.0) * k1 - (355.0 / 33.0) * k2 + (46732.0 / 5247.0) * k3 + (49.0 / 176.0) * k4 - (5103.0 / 18656.0) * k5), a, b)
        yn = y + h * ((35.0 / 384.0) * k1 + (500.0 / 1113.0) * k3 + (125.0 / 192.0) * k4 - (2187.0 / 6784.0) * k5 + (11.0 / 84.0) * k6)
        k7 = rhs(yn, a, b)
        e = h * ((71.0 / 57600.0) * k1 - (71.0 / 16695.0) * k3 + (71.0 / 1920.0) * k4 - (17253.0 / 339200.0) * k5 + (22.0 / 525.0) * k6 - (1.0 / 40.0) * k7)
        err = float(0.0)
        for j in range(5):
            sc = atol + rtol * wp.max(wp.abs(y[j]), wp.abs(yn[j]))
            err = wp.max(err, wp.abs(e[j]) / sc)
        # reject steps that leave the physical domain or produce non-finite values
        # (note: NaN compares false and fmax(err, NaN) == err, so test explicitly)
        chk = yn[0] + yn[1] + yn[2] + yn[3] + yn[4] + err
        if wp.isnan(chk) or wp.isinf(chk):
            h = h * 0.1
            if h < 1.0e-12:
                st = 3
            continue
        if yn[0] < 0.5 * r_stop:
            err = wp.max(err, 2.0)
        if err > 1.0 and h > 1.0e-9:
            h = h * wp.max(0.2, 0.9 * wp.pow(err, -0.2))
            continue
        # --- accepted step: check equatorial crossing
        c0 = wp.cos(y[1])
        c1 = wp.cos(yn[1])
        if c0 * c1 < 0.0 and nc < MAX_CROSS:
            # target theta = pi/2 + k pi lying between y[1] and yn[1]
            tgt = 1.5707963 + 3.14159265 * wp.floor((wp.max(y[1], yn[1]) - 1.5707963) / 3.14159265)
            t = (tgt - y[1]) / (yn[1] - y[1])
            t = wp.clamp(t, 0.0, 1.0)
            for it in range(3):
                f = hermite(y[1], yn[1], k1[1], k7[1], h, t) - tgt
                df = hermite_d(y[1], yn[1], k1[1], k7[1], h, t)
                if wp.abs(df) > 1.0e-12:
                    t = wp.clamp(t - f / df, 0.0, 1.0)
            rc = hermite(y[0], yn[0], k1[0], k7[0], h, t)
            phc = hermite(y[2], yn[2], k1[2], k7[2], h, t)
            prc = y[3] + t * (yn[3] - y[3])
            # sign of d(theta)/dl at the crossing: +1 going north->south
            sgn = wp.sign(y[4] + t * (yn[4] - y[4]))
            cross[tid, nc] = wp.vec4(rc, phc, prc, sgn)
            nc += 1
            if rc >= disk_rin and rc <= disk_rout:
                st = 2
        y = yn
        k1 = k7
        # --- null-constraint projection.  In FP32 the impact parameter of a distant ray is
        # encoded in p_r ~ -1 only at the 1e-5 level, so rounding violates H~ = 0 by
        # O(r^2 * 1e-7).  b and (th, p_th) carry the exact conserved quantities, so we
        # re-solve H~ = 0 for p_r (skipping near radial turning points).
        s = wp.sin(y[1])
        if wp.abs(s) > 1.0e-4:
            rr = y[0]
            D = rr * rr - 2.0 * rr + a * a
            P = rr * rr + a * a - a * b
            W = -P * P / D + b * b / (s * s) - 2.0 * a * b + a * a * s * s
            rad = -(y[4] * y[4] + W) / D
            if rad > 0.0:
                prn = wp.sign(y[3]) * wp.sqrt(rad)
                if wp.abs(prn - y[3]) < 0.05 * wp.abs(y[3]):
                    y = vec5(y[0], y[1], y[2], prn, y[4])
                    k1 = rhs(y, a, b)
        h = h * wp.min(5.0, 0.9 * wp.pow(wp.max(err, 1.0e-10), -0.2))
        if st < 0:
            if y[0] < r_stop:
                st = 1
            elif y[0] > r_esc and y[3] < 0.0:  # backward ray receding (photon p_r < 0)
                st = 0
                # asymptotic direction of motion (flat space at large r)
                r = y[0]
                th = y[1]
                ph = y[2]
                vr = k1[0]
                vth = r * k1[1]
                vph = r * wp.sin(th) * k1[2]
                st_ = wp.sin(th)
                ct = wp.cos(th)
                sp = wp.sin(ph)
                cp = wp.cos(ph)
                vx = vr * st_ * cp + vth * ct * cp - vph * sp
                vy = vr * st_ * sp + vth * ct * sp + vph * cp
                vz = vr * ct - vth * st_
                esc_dir[tid] = wp.normalize(wp.vec3(vx, vy, vz))
    if st < 0:
        st = 3
    status[tid] = st
    ncross[tid] = nc
    nsteps[tid] = i


# ----------------------------------------------------------------------------- camera


@dataclass
class Camera:
    """A camera at Boyer-Lindquist position (r, theta, phi).

    Orientation vectors are given in the camera's local orthonormal frame, whose
    axes coincide with the ZAMO (locally non-rotating) frame axes (e_r, e_theta,
    e_phi).  `beta` is the camera's 3-velocity relative to the ZAMO, in the same
    basis (zero = a ZAMO camera).  fov is the horizontal field of view in degrees.
    """

    r: float
    theta: float
    phi: float = 0.0
    forward: tuple = (-1.0, 0.0, 0.0)  # look at the hole
    up: tuple = (0.0, -1.0, 0.0)  # towards the north pole
    fov: float = 30.0
    beta: tuple = (0.0, 0.0, 0.0)


def _normalize(v):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)


def camera_rays(cam: Camera, a: float, width: int, height: int, ss: int = 1, device="cuda", jitter: bool = False):
    """Initial conditions (p_r, p_theta, b) with E = 1 for each (supersampled) pixel.

    Returns tensors of shape (height*ss, width*ss) and the camera-frame photon
    energy E_cam (for E = 1 normalisation), which sets every redshift factor.
    """
    W, H = width * ss, height * ss
    f = torch.tensor(cam.forward, dtype=torch.float64, device=device)
    u = torch.tensor(cam.up, dtype=torch.float64, device=device)
    f = f / f.norm()
    rgt = torch.linalg.cross(f, u)
    rgt = rgt / rgt.norm()
    u = torch.linalg.cross(rgt, f)
    tx = math.tan(math.radians(cam.fov) / 2)
    ty = tx * H / W
    xs = (torch.arange(W, device=device, dtype=torch.float64) + 0.5) / W * 2 - 1
    ys = 1 - (torch.arange(H, device=device, dtype=torch.float64) + 0.5) / H * 2
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    if jitter:
        X = X + (torch.rand_like(X) - 0.5) * 2 / W
        Y = Y + (torch.rand_like(Y) - 0.5) * 2 / H
    n = f[None, None, :] + X[..., None] * tx * rgt[None, None, :] + Y[..., None] * ty * u[None, None, :]
    n = _normalize(n)  # direction the camera looks, in (r, th, ph) frame components
    # photon 4-momentum in the camera frame (photon travels towards the camera: -n)
    p0 = torch.ones_like(X)
    pv = -n
    beta = torch.tensor(cam.beta, dtype=torch.float64, device=device)
    bmag = float(beta.norm())
    if bmag > 0:
        g = 1.0 / math.sqrt(1 - bmag * bmag)
        bh = beta / bmag
        bp = (pv * bh).sum(-1)
        p0F = g * (p0 + bmag * bp)
        pvF = pv + ((g - 1) * bp + g * bmag * p0)[..., None] * bh
    else:
        p0F, pvF = p0, pv
    # ZAMO frame -> covariant Boyer-Lindquist components
    r, th = cam.r, cam.theta
    st, ct = math.sin(th), math.cos(th)
    rho2 = r * r + a * a * ct * ct
    D = r * r - 2 * r + a * a
    Sig2 = (r * r + a * a) ** 2 - a * a * D * st * st
    alpha = math.sqrt(rho2 * D / Sig2)
    omega = 2 * a * r / Sig2
    varpi = math.sqrt(Sig2 / rho2) * st
    p_t = -(alpha * p0F + omega * varpi * pvF[..., 2])
    p_r = math.sqrt(rho2 / D) * pvF[..., 0]
    p_th = math.sqrt(rho2) * pvF[..., 1]
    p_ph = varpi * pvF[..., 2]
    E = -p_t
    out = dict(
        pr=(p_r / E).float(),
        pth=(p_th / E).float(),
        b=(p_ph / E).float(),
        E_cam=(1.0 / E).float(),  # camera-frame energy when E=1
        n_cam=n.float(),
    )
    return out


# ----------------------------------------------------------------------------- driver


@dataclass
class TraceResult:
    status: torch.Tensor  # (H, W) int32
    esc_dir: torch.Tensor  # (H, W, 3)
    cross: torch.Tensor  # (H, W, MAX_CROSS, 4): r, phi, p_r, sign(dth)
    ncross: torch.Tensor  # (H, W)
    nsteps: torch.Tensor  # (H, W)
    b: torch.Tensor  # (H, W)
    E_cam: torch.Tensor  # (H, W)
    a: float


def stop_radius(a: float) -> float:
    """Rays reaching this radius are counted as captured (just outside r_+)."""
    r_h = horizon_radius(a)
    return r_h + 0.01 * (2.0 - r_h) + 1e-3


def trace_rays(r0, th0, ph0, pr, pth, b, a, disk_rin=1e9, disk_rout=-1e9, r_esc=2000.0,
               max_steps=4000, rtol=1e-5, atol=1e-6, cap_scale=1.0):
    """Low-level: trace rays from a common start point with given (p_r, p_th, b), E = 1.

    pr/pth/b are flat float32 CUDA tensors. Returns a dict of flat torch tensors.
    """
    n = pr.numel()
    status = wp.zeros(n, dtype=wp.int32)
    esc = wp.zeros(n, dtype=wp.vec3)
    cross = wp.zeros((n, MAX_CROSS), dtype=wp.vec4)
    ncross = wp.zeros(n, dtype=wp.int32)
    nsteps = wp.zeros(n, dtype=wp.int32)
    wp.launch(
        trace_kernel,
        dim=n,
        inputs=[wp.from_torch(pr.contiguous()), wp.from_torch(pth.contiguous()), wp.from_torch(b.contiguous()),
                float(r0), float(th0), float(ph0), float(a), float(stop_radius(a)), float(r_esc),
                float(disk_rin), float(disk_rout), int(max_steps), float(rtol), float(atol), float(cap_scale)],
        outputs=[status, esc, cross, ncross, nsteps],
    )
    wp.synchronize_device()  # Warp and torch may use different streams
    return dict(
        status=wp.to_torch(status),
        esc_dir=wp.to_torch(esc),
        cross=wp.to_torch(cross),
        ncross=wp.to_torch(ncross),
        nsteps=wp.to_torch(nsteps),
    )


def trace(
    cam: Camera,
    a: float,
    width: int,
    height: int,
    ss: int = 1,
    disk_rin: float = 1e9,
    disk_rout: float = -1e9,
    r_esc: float | None = None,
    max_steps: int = 4000,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    jitter: bool = False,
) -> TraceResult:
    init = camera_rays(cam, a, width, height, ss, jitter=jitter)
    Hs, Ws = init["b"].shape
    if r_esc is None:
        r_esc = max(2000.0, 4.0 * cam.r)
    out = trace_rays(cam.r, cam.theta, cam.phi, init["pr"].reshape(-1), init["pth"].reshape(-1),
                     init["b"].reshape(-1), a, disk_rin, disk_rout, r_esc, max_steps, rtol, atol)
    return TraceResult(
        status=out["status"].reshape(Hs, Ws),
        esc_dir=out["esc_dir"].reshape(Hs, Ws, 3),
        cross=out["cross"].reshape(Hs, Ws, MAX_CROSS, 4),
        ncross=out["ncross"].reshape(Hs, Ws),
        nsteps=out["nsteps"].reshape(Hs, Ws),
        b=init["b"],
        E_cam=init["E_cam"],
        a=a,
    )
