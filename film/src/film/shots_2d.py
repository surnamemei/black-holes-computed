"""2D explanatory shots: light-ray diagram, scale, EHT globe, Fourier, reconstruction, finale, credits."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .core import (AMBER, BG, BLUE, CYAN, DEV, FPS, GREY, H, RED, W, WHITE, Canvas, Shot, ease_in_out, ease_out,
                   layout_cues, lerp, loglerp, over, ramp, smooth, smoother, vignette_bg, window)

# ============================================================================= 3. light rays


def schwarzschild_ray(b, x0=-30.0, xmax=40.0, M=1.0, ds=0.004, max_len=400.0):
    """Exact Schwarzschild null orbit shape via d2r/dl2 = -3 M L^2 r / r^5 (u'' + u = 3 M u^2),
    launched from (x0, b) moving in +x.  Returns (N, 2) points and whether it was captured."""
    p = np.array([x0, b], float)
    v = np.array([1.0, 0.0])
    L2 = b * b
    pts = [p.copy()]
    s = 0.0
    acc = lambda q: -3 * M * L2 * q / np.dot(q, q) ** 2.5
    while s < max_len:
        r = math.hypot(*p)
        h = ds * max(1.0, min(r - 1.5, 10.0)) * (0.25 if r < 4 else 1.0)
        # RK4 on (p, v)
        k1p, k1v = v, acc(p)
        k2p, k2v = v + 0.5 * h * k1v, acc(p + 0.5 * h * k1p)
        k3p, k3v = v + 0.5 * h * k2v, acc(p + 0.5 * h * k2p)
        k4p, k4v = v + h * k3v, acc(p + h * k3p)
        p = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        s += h
        pts.append(p.copy())
        if math.hypot(*p) < 2.0 * M:
            return np.array(pts), True
        if p[0] > xmax or abs(p[1]) > 30 or (p[0] < x0 - 1):
            return np.array(pts), False
    return np.array(pts), False


def arclen(pts):
    d = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
    return np.concatenate([[0], np.cumsum(d)])


class RaysShot(Shot):
    name = "rays"
    SCALE = 32.0  # px per M
    CX, CY = W * 0.56, H * 0.52

    def __init__(self):
        self.cues, self.duration = layout_cues(["rays1", "rays2", "rays3"], lead=0.8, gap=0.9, tail=1.4)
        c1, c2, c3 = self.cues
        bc = math.sqrt(27)
        self.groups = [  # (launch time, list of b, colour)
            (c1.word("far") - 0.6, [15.5, 12.0], AMBER),
            (c1.word("Closer") - 0.2, [7.5, 6.2, 5.35], AMBER),
            (c2.word("circle") - 1.0, [bc * (1 + 2e-9)], WHITE),
            (c2.word("closer") - 0.2, [4.8, 3.6, 2.2], RED),
        ]
        self.t_sphere = c2.word("photon")
        self.t_horizon = c1.start + 0.3
        self.t_shadow = c3.start
        self.t_wider = c3.word("wider") - 0.8
        self.c3 = c3

    def setup(self):
        self.rays = []
        for t0, bs, col in self.groups:
            for k, b in enumerate(bs):
                pts, cap = schwarzschild_ray(b, x0=-34.0, max_len=170 if col is WHITE else 130)
                self.rays.append(dict(t0=t0 + 0.3 * k, pts=pts, s=arclen(pts), col=col, cap=cap))
        # parallel beam for the shadow explanation (both signs of b)
        self.fan = []
        for b in np.linspace(-11, 11, 31):
            pts, cap = schwarzschild_ray(abs(b) + 1e-6, x0=-34.0, max_len=90)
            if not cap:  # keep the right side calm: stop escaping rays ~14 M past the hole
                L = arclen(pts)
                k = np.searchsorted(L, 34.0 + 14.0 + 6.0 * (6.0 / max(abs(b), 5.3)))
                pts = pts[:max(k, 2)]
            if b < 0:
                pts = pts * np.array([1, -1])
            self.fan.append(dict(pts=pts, s=arclen(pts), cap=cap, b=b))

    def to_px(self, pts):
        return [(self.CX + x * self.SCALE, self.CY - y * self.SCALE) for x, y in pts]

    def frame(self, t):
        img = vignette_bg()
        cv = Canvas()
        S, cx, cy = self.SCALE, self.CX, self.CY
        a_main = 1 - ramp(t, self.t_shadow - 0.2, self.t_shadow + 1.0)
        speed = 24.0  # M per second along each ray
        for ray in self.rays:
            if t < ray["t0"] or a_main <= 0:
                continue
            sp = speed * (2.4 if ray["col"] is WHITE else 1.0)
            L = (t - ray["t0"]) * sp
            n = int(np.searchsorted(ray["s"], L))
            if n < 2:
                continue
            pts = self.to_px(ray["pts"][:n:2] if n > 6 else ray["pts"][:n])
            col = ray["col"]
            cv.polyline(pts, col, 2.2 if col is WHITE else 1.8, 0.95 * a_main, glow=3, glow_alpha=0.3)
            if n < len(ray["s"]) - 1:
                cv.circle(*pts[-1], 4, fill=WHITE, alpha=a_main, glow=6)
        # --- shadow explanation: a parallel beam; everything inside +-sqrt(27) M is captured
        a_fan = ramp(t, self.t_shadow + 0.2, self.t_shadow + 1.2)
        bc = math.sqrt(27)
        if a_fan > 0:
            prog = ramp(t, self.t_shadow + 0.2, self.t_shadow + 4.0, ease_in_out)
            for ray in self.fan:
                n = max(2, int(len(ray["pts"]) * prog))
                col = RED if ray["cap"] else AMBER
                pts = self.to_px(ray["pts"][:n:3])
                cv.polyline(pts, col, 1.2, (0.55 if ray["cap"] else 0.7) * a_fan)
            x_l = cx - 27 * S
            a_br = ramp(t, self.t_wider - 1.4, self.t_wider - 0.2)
            yb0, yb1 = cy - bc * S, cy + bc * S
            cv.rect(x_l - 60, yb0, 120, yb1 - yb0, fill=RED, alpha=0.10 * a_fan)
            cv.text("every ray in this band", x_l, yb0 - 46, 24, "Regular", WHITE, a_fan, "center", shadow=6)
            cv.text("falls in", x_l, yb0 - 18, 24, "Regular", WHITE, a_fan, "center", shadow=6)
            if a_br > 0:
                xs, xh = x_l + 90, x_l + 120
                cv.polyline([(xs - 12, yb0), (xs, yb0), (xs, yb1), (xs - 12, yb1)], WHITE, 2.2, a_br)
                cv.text("shadow", xs + 14, cy + 9, 24, "Medium", WHITE, a_br, shadow=6)
                cv.polyline([(cx - 2 * S, cy + 2 * S + 26), (cx - 2 * S, cy + 2 * S + 40), (cx + 2 * S, cy + 2 * S + 40),
                             (cx + 2 * S, cy + 2 * S + 26)], GREY, 2, a_br)
                cv.text("horizon", cx, cy + 2 * S + 70, 22, "Regular", GREY, a_br, "center", shadow=6)
                cv.text("shadow ≈ 2.6 × horizon", W * 0.5, H * 0.92, 40, "Medium", WHITE, a_br, "center", shadow=8)
        # photon sphere (drawn above the rays)
        a_ps = ramp(t, self.t_sphere - 0.4, self.t_sphere + 0.6) * (1 - 0.5 * ramp(t, self.t_shadow, self.t_shadow + 1))
        if a_ps > 0:
            cv.circle(cx, cy, 3 * S, stroke=CYAN, width=2, alpha=0.9 * a_ps, dash=(9, 8))
            ax, ay = cx + 3 * S * 0.71, cy - 3 * S * 0.71
            cv.polyline([(ax, ay), (ax + 70, ay - 70), (ax + 90, ay - 70)], CYAN, 1.5, a_ps)
            cv.text("photon sphere", ax + 98, ay - 62, 30, "Medium", CYAN, a_ps, shadow=6)
            cv.text("1.5 × horizon radius", ax + 98, ay - 30, 22, "Regular", GREY, a_ps, shadow=6)
        # event horizon
        a_h = ramp(t, 0.0, 1.0)
        cv.circle(cx, cy, 2 * S + 5, fill=(0.0, 0.0, 0.0), alpha=0.6 * a_h, glow=8)
        cv.circle(cx, cy, 2 * S, fill=(0.0, 0.0, 0.0), alpha=a_h)
        cv.circle(cx, cy, 2 * S, stroke=(0.35, 0.38, 0.45), width=1.5, alpha=a_h)
        a_hl = window(t, self.t_horizon, self.t_sphere - 0.5, 0.8, 0.8)
        cv.text("event horizon", cx, cy + 2 * S + 40, 24, "Medium", WHITE, a_hl, "center", shadow=6)
        return over(img, cv.to_tensor())


def extend(tl):
    """Insert the 2D shots into the timeline (called by timeline.build)."""
    out = []
    for shot, xf in tl:
        out.append((shot, xf))
        if shot.name == "lensing":
            out.append((RaysShot(), 1.0))
    for cls, xf in _TAIL:
        out.append((cls(), xf))
    return out


_TAIL = []  # filled at the end of the module


# ============================================================================= 8. EHT globe

from .globe import Globe  # noqa: E402

SITES = [  # narration word, station names, label
    ("Hawaii", ["JCMT", "SMA"], "JCMT + SMA · Hawaii"),
    ("Arizona", ["SMT"], "SMT · Arizona"),
    ("Mexico", ["LMT"], "LMT · Mexico"),
    ("Chile", ["ALMA", "APEX"], "ALMA + APEX · Chile"),
    ("Spain", ["IRAM"], "IRAM 30 m · Spain"),
    ("South", ["SPT"], "SPT · South Pole"),
]
CORRELATORS = [("MIT Haystack", 42.62, -71.49), ("MPIfR Bonn", 50.73, 7.09)]


def station_table():
    from src.vlbi.eht import STATIONS
    return {s[0]: (s[2], s[3]) for s in STATIONS}


def slerp_path(a, b, n=64, lift=0.18):
    """Great-circle arc between unit vectors a and b, lifted above the surface (for flights)."""
    om = math.acos(np.clip(a @ b, -1, 1))
    ts = np.linspace(0, 1, n)
    pts = np.array([(math.sin((1 - t) * om) * a + math.sin(t * om) * b) / max(math.sin(om), 1e-9) for t in ts])
    return pts * (1 + lift * np.sin(np.pi * ts))[:, None]


def ll2v(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


class EHTShot(Shot):
    name = "eht"

    def __init__(self):
        self.cues, self.duration = layout_cues(["eht1", "eht2", "eht3"], lead=0.8, gap=0.8, tail=1.0)
        c1, c2, c3 = self.cues
        self.t_missing = c1.word("missing")
        self.t_sites = [c2.word(w) for w, _, _ in SITES]
        self.t_clock = c2.word("atomic")
        self.t_flown = c3.word("flown") - 0.3
        self.t_pairs = c3.word("pair") - 0.6
        self.views = [(8, -88)] + [(20, -140), (28, -112), (16, -100), (-14, -74), (26, -42), (-35, -70)] + [(2, -88)]
        self.view_t = [0.0] + [t - 0.6 for t in self.t_sites] + [c2.end - 0.5]

    def setup(self):
        self.globe = Globe()
        self.st = station_table()

    def view(self, t):
        vt, vv = self.view_t, self.views
        if t <= vt[0]:
            return vv[0]
        for k in range(len(vt) - 1):
            if t < vt[k + 1]:
                x = ease_in_out((t - vt[k]) / max(vt[k + 1] - vt[k], 1e-3) * 1.4)
                return (lerp(vv[k][0], vv[k + 1][0], x), lerp(vv[k][1], vv[k + 1][1], x))
        return vv[-1]

    def frame(self, t):
        R, cx, cy = 400.0, W * 0.5, H * 0.53
        lat0, lon0 = self.view(t)
        img = vignette_bg()
        g, cov = self.globe.render(lat0, lon0, R, cx, cy)
        img = img * (1 - cov) + g
        cv = Canvas()
        # the impossible Earth-sized dish, which then goes "mostly missing"
        a_dish = window(t, 0.6, self.t_missing + 1.4, 1.0, 1.2)
        if a_dish > 0:
            for k in range(1, 7):
                cv.circle(cx, cy, R * k / 6, stroke=CYAN, width=1.4, alpha=0.45 * a_dish)
            for k in range(16):
                a_ = 2 * math.pi * k / 16
                cv.polyline([(cx, cy), (cx + R * math.cos(a_), cy + R * math.sin(a_))], CYAN, 1.2, 0.35 * a_dish)
            cv.circle(cx, cy, R, stroke=CYAN, width=3, alpha=0.8 * a_dish, glow=6)
        # stations
        site_pos = {}
        for k, (word, names, lab) in enumerate(SITES):
            ts = self.t_sites[k]
            a_st = ramp(t, ts - 0.2, ts + 0.4) if t < self.t_missing + 0.5 or True else 0
            a_pre = ramp(t, self.t_missing - 0.1, self.t_missing + 0.6)  # pieces left behind by the dish
            a = max(a_st, 0.55 * a_pre)
            if a <= 0:
                continue
            for nm in names:
                lat, lon = self.st[nm]
                x, y, z = self.globe.project(lat, lon, lat0, lon0, R, cx, cy)
                site_pos[nm] = (x, y, z)
            lat, lon = self.st[names[0]]
            x, y, z = self.globe.project(lat, lon, lat0, lon0, R, cx, cy)
            vis = smooth((z + 0.05) / 0.15)
            if vis <= 0:
                continue
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * (t - ts) / 1.2)
            cv.circle(x, y, 7, fill=AMBER, alpha=a * vis, glow=8)
            if t > ts:
                cv.circle(x, y, 10 + 16 * ((t - ts) % 1.2) / 1.2, stroke=AMBER, width=1.5, alpha=a_st * vis * (1 - ((t - ts) % 1.2) / 1.2))
            a_lab = a_st * window(t, ts, self.t_flown, 0.4, 0.6) * vis
            if a_lab > 0:
                dx = 1 if x > cx - 60 else -1
                cv.text(lab, x + dx * 22, y + 8, 26, "Medium", WHITE, a_lab, "left" if dx > 0 else "right", shadow=8)
        a_clk = window(t, self.t_clock - 0.3, self.t_flown, 0.5, 0.6)
        if a_clk > 0:
            cv.text("every station: hydrogen-maser atomic clock", W / 2, H - 70, 28, "Regular", GREY, a_clk, "center", shadow=8)
        # hard drives flown to the correlators
        a_fl = window(t, self.t_flown, self.duration + 1, 0.4, 0.6)
        if a_fl > 0:
            prog = ramp(t, self.t_flown, self.t_flown + 3.0, ease_in_out)
            for name, la_c, lo_c in CORRELATORS:
                x, y, z = self.globe.project(la_c, lo_c, lat0, lon0, R, cx, cy)
                if z > 0:
                    cv.rect(x - 7, y - 7, 14, 14, fill=CYAN, alpha=a_fl, radius=2)
                    cv.text(name, x + 16, y - 12, 24, "Medium", CYAN, a_fl * (1 - ramp(t, self.t_pairs, self.t_pairs + 0.6)), shadow=8)
            for k, (word, names, lab) in enumerate(SITES):
                lat, lon = self.st[names[0]]
                tgt = CORRELATORS[0] if lon < -40 else CORRELATORS[1]
                path = slerp_path(ll2v(lat, lon), ll2v(tgt[1], tgt[2]), 48, 0.12)
                P = self.globe.project_xyz(path, lat0, lon0, R, cx, cy)
                n = max(2, int(len(P) * prog))
                seg = [(p[0], p[1]) for p in P[:n] if p[2] > -0.05 or np.hypot(p[0] - cx, p[1] - cy) > R]
                cv.polyline(seg, CYAN, 2.4, 0.95 * a_fl * (1 - 0.6 * ramp(t, self.t_pairs, self.t_pairs + 0.6)), glow=3, glow_alpha=0.4)
                if 0 < prog < 1 and seg:
                    cv.circle(*seg[-1], 4.5, fill=WHITE, alpha=a_fl, glow=5)
        # every pair of telescopes: baselines (chords through the Earth)
        a_bl = ramp(t, self.t_pairs, self.t_pairs + 1.2)
        if a_bl > 0:
            names = [n for _, ns, _ in SITES for n in ns]
            pos = {n: self.globe.project(*self.st[n], lat0, lon0, R, cx, cy) for n in names}
            k = 0
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    (x1, y1, z1), (x2, y2, z2) = pos[names[i]], pos[names[j]]
                    d = ramp(t, self.t_pairs + 0.05 * k, self.t_pairs + 0.05 * k + 0.5)
                    if d > 0:
                        cv.polyline([(x1, y1), (lerp(x1, x2, d), lerp(y1, y2, d))], CYAN, 2.2, 0.9 * a_bl, glow=4, glow_alpha=0.45)
                    k += 1
            cv.text("28 baselines", W / 2, H - 70, 30, "Medium", CYAN, a_bl, "center", shadow=8)
        a_t = window(t, 0.3, self.t_sites[0] - 0.3, 0.6, 0.6)
        if a_t > 0 and t > self.t_missing - 0.2:
            cv.text("a telescope the size of the Earth", W / 2, 110, 34, "Light", WHITE, a_t, "center", shadow=8)
        return over(img, cv.to_tensor())




# ============================================================================= shared helpers for data panels

import os as _os  # noqa: E402

from .core import ROOT  # noqa: E402

_SIM = None


def sim():
    global _SIM
    if _SIM is None:
        _SIM = dict(np.load(_os.path.join(ROOT, "out", "vlbi", "sim.npz")))
    return _SIM


def afmhot_t(x):
    x = x.clamp(0, 1)
    return torch.stack([(2 * x).clamp(0, 1), (2 * x - 0.5).clamp(0, 1), (2 * x - 1).clamp(0, 1)], 0)


def icefire(x):
    """dark blue -> cyan -> white map for Fourier amplitudes."""
    x = x.clamp(0, 1)
    r = (x * 1.6 - 0.75).clamp(0, 1)
    g = (x * 1.3 - 0.2).clamp(0, 1)
    b = (0.25 + x * 1.2).clamp(0, 1)
    return torch.stack([r, g, b], 0)


def panel_image(img, size, cmap=afmhot_t, gamma=0.7, norm=None):
    """(n,n) numpy/tensor -> (3,size,size) tensor, bicubic-upsampled, colour-mapped."""
    t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device=DEV)[None, None]
    t = Fn.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)[0, 0].clamp(min=0)
    m = norm if norm is not None else float(t.max()) + 1e-12
    return cmap((t / m) ** gamma)


def paste(img, tile, x, y, alpha=1.0):
    """Paste a (3,h,w) tile into img at top-left (x, y) with alpha (clipped to the frame)."""
    _, h, w = tile.shape
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return img
    img[:, y0:y1, x0:x1] = img[:, y0:y1, x0:x1] * (1 - alpha) + tile[:, y0 - y:y1 - y, x0 - x:x1 - x] * alpha
    return img


def frame_box(cv, x, y, w, h, alpha, color=(0.3, 0.36, 0.45)):
    cv.rect(x - 1, y - 1, w + 2, h + 2, stroke=color, width=1.5, alpha=alpha)


def grating(size, kx, ky, phase=0.0):
    """Ripple cos(2 pi (kx x + ky y) + phase) on [-0.5, 0.5]^2, mapped to [0, 1]."""
    c = torch.linspace(-0.5, 0.5, size, device=DEV)
    Y, X = torch.meshgrid(-c, c, indexing="ij")
    return 0.5 + 0.5 * torch.cos(2 * math.pi * (kx * X + ky * Y) + phase)


def dish_icon(cv, x, y, s, color=WHITE, alpha=1.0, angle=-90):
    """A radio dish seen from the side (filled reflector, feed legs, mount), pointing at `angle` deg."""
    import skia
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)  # boresight
    px, py = -uy, ux
    P = lambda u, d: (x + (u * px + d * ux) * s, y + (u * py + d * uy) * s)
    path = skia.Path()
    for k in range(25):
        u = (k / 24 - 0.5) * 2
        pt = P(u, 0.38 * u * u)
        path.moveTo(*pt) if k == 0 else path.lineTo(*pt)
    for k in range(25):
        u = (0.5 - k / 24) * 2 * 0.96
        path.lineTo(*P(u, 0.38 * u * u + 0.10))
    path.close()
    paint = skia.Paint(AntiAlias=True)
    paint.setColor4f(skia.Color4f(color[0], color[1], color[2], alpha * 0.9))
    cv.c.drawPath(path, paint)
    tip = P(0, 0.62)
    for u in (-0.85, 0.85):
        cv.polyline([P(u, 0.38 * u * u), tip], color, 1.4, alpha * 0.8)
    cv.circle(*tip, max(2.0, s * 0.05), fill=color, alpha=alpha)
    base = P(0, -0.45)
    cv.polyline([P(0, 0.08), base], color, max(2.0, s * 0.07), alpha)
    cv.polyline([P(-0.35, -0.45), P(0.35, -0.45)], color, max(2.0, s * 0.06), alpha)


# ============================================================================= 7. scale


PLANETS = [("Mercury", 0.387), ("Venus", 0.723), ("Earth", 1.0), ("Mars", 1.524), ("Jupiter", 5.20),
           ("Saturn", 9.58), ("Uranus", 19.2), ("Neptune", 30.1)]
R_S_AU = 128.3  # Schwarzschild radius of 6.5e9 Msun in AU


def moon_surface(zoom, R, cx, cy):
    """Procedural full Moon whose detail is defined per *pixel scale*, so it stays sharp and
    statistically self-similar through a zoom of millions."""
    from src.kerr.sky import value_noise3
    yy, xx = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float64) + 0.5,
                            torch.arange(W, device=DEV, dtype=torch.float64) + 0.5, indexing="ij")
    qx = (xx - cx) / (R * zoom)
    qy = (cy - yy) / (R * zoom)
    r2 = qx * qx + qy * qy
    z = torch.sqrt(torch.clamp(1 - r2, min=0))
    px = 1.0 / (R * zoom)  # Moon radii per pixel
    alb = torch.zeros_like(qx)
    wsum = 0.0
    for j in range(40):
        f = 2.0 * 2.0 ** j
        c = f * px  # cycles per pixel
        if c > 0.5:
            break
        if c < 1 / 4000:
            continue
        w = c ** -0.38 * min(1.0, math.log2(0.5 / c) / 1.5) * min(1.0, math.log2(c * 4000) / 2.0)
        # positive offset (not a modulo): wrapping negative coordinates would leave seams at q = 0
        p = torch.stack([qx * f + 517.3 + 31.1 * j, qy * f + 733.9 + 17.7 * j, torch.full_like(qx, 13.7 * j)], -1).float()
        n = value_noise3(p, seed=40 + j).double()
        alb = alb + w * (n - 0.5)
        wsum += w
    alb = alb / max(wsum, 1e-9) * 2.2
    # maria: dark plains, only meaningful at low zoom
    m = value_noise3(torch.stack([qx * 3.2 + 0.4, qy * 3.2 - 0.7, torch.full_like(qx, 5.0)], -1).float(), seed=7).double()
    m2 = value_noise3(torch.stack([qx * 7.0, qy * 7.0, torch.full_like(qx, 9.0)], -1).float(), seed=8).double()
    mar = torch.clamp((0.55 * m + 0.45 * m2 - 0.5) * 6, 0, 1) * max(0.0, 1 - math.log10(zoom) / 2)
    bright = 0.66 - 0.26 * mar + 0.30 * alb
    shade = (0.6 + 0.4 * z ** 0.5)
    edge = torch.clamp((1 - torch.sqrt(r2)) * R * zoom / 1.2, 0, 1)
    v = (bright * shade).clamp(0, 1) * edge
    col = torch.stack([v * 0.97, v * 0.96, v * 0.93], 0).float()
    return col, edge.float()[None]


class ScaleShot(Shot):
    name = "scale"

    def __init__(self):
        self.cues, self.duration = layout_cues(["scale1", "scale2", "scale3"], lead=1.0, gap=0.9, tail=1.2)
        c1, c2, c3 = self.cues
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.t_moon = c2.start - 0.6
        self.t_zoom0 = c2.word("orange") - 0.4
        self.t_zoom1 = c2.word("Earth") + 0.3
        self.tC = c3.start - 0.5
        self.t_div = c3.word("divided")
        self.t_13 = c3.word("one")
        self.t_earth = c3.word("Earth") - 2.2

    def frame(self, t):
        img = vignette_bg()
        cv = Canvas()
        cx, cy = W / 2, H / 2
        # ---------------- A: the solar system fits inside the horizon
        aA = 1 - ramp(t, self.t_moon - 0.3, self.t_moon + 0.5)
        if aA > 0:
            s = ease_in_out((t - 0.3) / (self.c1.end - 0.3))
            ppa = loglerp(240, 1.85, s)
            rh = ppa * R_S_AU
            if rh < 6000:
                cv.circle(cx, cy, rh, stroke=AMBER, width=3, alpha=aA, glow=10)
                a_h = aA * (1 - ramp(rh, 500, 1400))
                cv.text("event horizon of M87*", cx + rh * 0.62 + 30, cy + rh * 0.78 + 10, 30, "Medium", AMBER, a_h, shadow=8)
                cv.text("radius ≈ 130 AU", cx + rh * 0.62 + 30, cy + rh * 0.78 + 44, 22, "Regular", GREY, a_h, shadow=6)
            for k, (nm, au) in enumerate(PLANETS):
                r = ppa * au
                if r < 3 or r > 3000:
                    continue
                a_o = aA * min(1.0, r / 12)
                cv.circle(cx, cy, r, stroke=(0.5, 0.56, 0.66), width=1.4, alpha=0.7 * a_o)
                ang = 0.6 + 0.9 * k + t * 0.6 / math.sqrt(au)
                cv.circle(cx + r * math.cos(ang), cy - r * math.sin(ang), 4, fill=WHITE, alpha=a_o)
                a_l = a_o * ramp(r, 25, 60) * (1 - ramp(r, 600, 900))
                if nm in ("Earth", "Jupiter", "Neptune") or (nm == "Saturn" and r > 200):
                    cv.text(nm, cx + r * math.cos(ang) + 10, cy - r * math.sin(ang) - 10, 22, "Regular", WHITE, a_l, shadow=6)
            cv.circle(cx, cy, 5, fill=(1.0, 0.85, 0.4), alpha=aA, glow=6)
            cv.text("our Solar System", cx, 120, 30, "Light", WHITE, aA * window(t, 0.8, self.c1.end, 0.8, 1.0), "center", shadow=8)
        # ---------------- B: 42 microarcseconds = an orange on the Moon
        aB = window(t, self.t_moon, self.tC + 0.6, 0.8, 0.8)
        if aB > 0:
            zx = ramp(t, self.t_zoom0, self.t_zoom1, ease_in_out)
            zoom = math.exp(zx * math.log(6.0e6))
            Rm = 380.0
            mc, cov = moon_surface(zoom, Rm, cx, cy)
            img = img * (1 - cov * aB) + mc * aB
            a_t = aB * (1 - ramp(t, self.t_zoom0, self.t_zoom0 + 0.6))
            cv.text("the full Moon: about 1,900,000,000 µas across", cx, H - 120, 30, "Regular", WHITE, a_t, "center", shadow=8)
            cv.text("M87*'s shadow: 42 µas", cx, H - 80, 30, "Medium", AMBER, a_t, "center", shadow=8)
            if zx > 0:
                cv.text(f"zoom × {zoom:,.0f}", 90, 100, 32, "Medium", WHITE, aB, shadow=8, backdrop=0.5)
            d_or = 2.25e-8 * 2 * Rm * zoom  # 7.8 cm on a 3474 km Moon
            if d_or > 1.5:
                r_ = d_or / 2
                cv.circle(cx + r_ * 0.25, cy + r_ * 0.3, r_ * 1.05, fill=(0, 0, 0), alpha=0.35 * aB, glow=r_ * 0.25 + 1)
                cv.circle(cx, cy, r_, fill=(0.95, 0.48, 0.08), alpha=aB)
                cv.circle(cx - r_ * 0.12, cy - r_ * 0.12, r_ * 0.78, fill=(1.0, 0.58, 0.14), alpha=aB)
                cv.circle(cx - r_ * 0.35, cy - r_ * 0.38, r_ * 0.22, fill=(1.0, 0.85, 0.6), alpha=0.6 * aB, glow=r_ * 0.12 + 1)
                cv.circle(cx + r_ * 0.05, cy - r_ * 0.93, max(r_ * 0.08, 1), fill=(0.3, 0.45, 0.15), alpha=aB)
                a_o = aB * ramp(d_or, 40, 120)
                if a_o > 0:
                    cv.polyline([(cx - r_, cy + r_ + 24), (cx - r_, cy + r_ + 34), (cx + r_, cy + r_ + 34), (cx + r_, cy + r_ + 24)], (0.1, 0.1, 0.1), 2.5, a_o)
                    cv.text("42 µas", cx, cy + r_ + 76, 36, "Medium", WHITE, a_o, "center", shadow=8, backdrop=0.6)
                    cv.text("an orange on the Moon, seen from Earth", cx, cy + r_ + 116, 24, "Regular", WHITE, a_o, "center", shadow=8, backdrop=0.6)
        # ---------------- C: resolution = wavelength / dish size
        aC = ramp(t, self.tC, self.tC + 0.8)
        if aC > 0:
            a_d = aC * (1 - ramp(t, self.t_earth, self.t_earth + 0.8))
            dx, dy, ds = W * 0.27, H * 0.50, 190
            if a_d > 0:
                dish_icon(cv, dx, dy, ds, WHITE, a_d, angle=0)
                cv.polyline([(dx + 0.35 * ds + 10, dy - ds), (dx + 0.35 * ds + 10, dy + ds)], AMBER, 2, a_d)
                cv.text("D", dx + 0.35 * ds + 24, dy + 10, 34, "Medium", AMBER, a_d, shadow=6)
                # incoming wave
                xs = np.linspace(dx + 120, W * 0.62, 300)
                lam = 70.0
                ph = t * 4.0
                pts = [(x, dy - 150 * 0 + 34 * math.sin(2 * math.pi * (x / lam) + ph) - 0) for x in xs]
                cv.polyline([(x, y - 190) for x, y in pts], CYAN, 2.2, a_d, glow=3, glow_alpha=0.3)
                x0 = xs[120]
                cv.polyline([(x0, dy - 250), (x0 + lam, dy - 250)], CYAN, 2, a_d)
                cv.text("λ", x0 + lam / 2, dy - 262, 32, "Medium", CYAN, a_d, "center", shadow=6)
                a_f = a_d * ramp(t, self.t_div - 0.3, self.t_div + 0.6)
                fx = W * 0.70
                cv.text("finest detail", fx, dy - 60, 30, "Regular", GREY, a_f, "center", shadow=6)
                cv.text("θ ≈ λ / D", fx, dy + 10, 64, "Light", WHITE, a_f, "center", shadow=8)
                a_n = a_d * ramp(t, self.t_13 - 0.3, self.t_13 + 0.6)
                cv.text("λ = 1.3 mm,  θ ≈ 20 µas", fx, dy + 90, 30, "Regular", WHITE, a_n, "center", shadow=6)
                a_n2 = a_d * ramp(t, self.t_13 + 1.4, self.t_13 + 2.2)
                cv.text("→  D ≈ 13,000 km", fx, dy + 140, 40, "Medium", AMBER, a_n2, "center", shadow=8)
            a_e = aC * ramp(t, self.t_earth, self.t_earth + 1.0)
            if a_e > 0:
                if not hasattr(self, "_globe"):
                    self._globe = Globe()
                ge, gc = self._globe.render(15, -60, 300, cx, cy)
                img = img * (1 - gc * a_e) + ge * a_e
                for k in range(1, 6):
                    cv.circle(cx, cy, 300 * k / 5, stroke=AMBER, width=1.2, alpha=0.35 * a_e)
                cv.circle(cx, cy, 300, stroke=AMBER, width=3, alpha=0.9 * a_e, glow=6)
                cv.polyline([(cx - 300, cy + 345), (cx + 300, cy + 345)], AMBER, 2, a_e)
                cv.text("a dish as wide as the Earth: 12,742 km", cx, cy + 395, 30, "Medium", WHITE, a_e, "center", shadow=8)
        return over(img, cv.to_tensor())


# ============================================================================= 9. Fourier


M87_DEC_DEG = 12.39
M87_RA_DEG = 187.706


def sub_source_lon(hours):
    from src.vlbi.eht import gmst_rad
    g = math.degrees(gmst_rad(2457854.5 + hours / 24.0))
    return ((M87_RA_DEG - g + 180) % 360) - 180


BL_COLORS = [(1.0, 0.71, 0.28), (0.33, 0.82, 1.0), (0.95, 0.45, 0.55), (0.55, 0.9, 0.5), (0.75, 0.6, 1.0),
             (1.0, 0.9, 0.4), (0.4, 0.95, 0.85)]


class FourierShot(Shot):
    name = "fourier"

    def __init__(self):
        self.cues, self.duration = layout_cues(["four1", "four2"], lead=0.6, gap=1.0, tail=1.6)
        c1, c2 = self.cues
        self.t_add = c1.word("Add") - 0.3
        self.t_ft = c1.word("Fourier") - 0.4
        self.t_far = c2.word("far") - 0.4
        self.t_turns = c2.word("Earth") - 0.8
        self.t_end = c2.end + 1.0

    def setup(self):
        d = sim()
        img = d["truth"]
        n = 128
        img = img.reshape(n, 2, n, 2).sum(axis=(1, 3))
        self.n = n
        F = np.fft.fft2(img)
        self.F = torch.tensor(F, dtype=torch.complex64, device=DEV)
        mag = np.abs(F).ravel()
        self.order = torch.tensor(np.argsort(-mag), device=DEV)
        self.img_max = float(img.max())
        self.uv = (d["u"], d["v"], d["t"], d["i"], d["j"])
        self.globe = Globe()
        from src.vlbi.eht import STATIONS
        self.stations = [(s[2], s[3]) for s in STATIONS]
        pairs = sorted(set(zip(d["i"].tolist(), d["j"].tolist())))
        self.pair_col = {p: BL_COLORS[k % len(BL_COLORS)] for k, p in enumerate(pairs)}

    def partial(self, N):
        mask = torch.zeros(self.n * self.n, device=DEV)
        mask[self.order[:N]] = 1
        rec = torch.fft.ifft2(self.F * mask.view(self.n, self.n)).real
        return rec, self.order[N - 1]

    def frame(self, t):
        img = vignette_bg()
        cv = Canvas()
        P = 420
        lx, rx, py = W * 0.30, W * 0.70, H * 0.47
        a1 = 1 - ramp(t, self.t_far - 0.5, self.t_far + 0.3)
        if a1 > 0:
            if t < self.t_add:
                # one baseline -> one ripple
                ph = t / self.t_add
                L = 0.35 + 0.65 * (0.5 - 0.5 * math.cos(2 * math.pi * ph * 1.2))
                ang = math.radians(10 + 110 * ph)
                bx, by = math.cos(ang), math.sin(ang)
                half = 170 * L
                for sgn in (-1, 1):
                    dish_icon(cv, lx + sgn * half * bx, py - sgn * half * by, 44, WHITE, a1, angle=-90)
                cv.polyline([(lx - half * bx, py + half * by), (lx + half * bx, py - half * by)], AMBER, 2, a1, dash=(8, 6))
                cv.text("two telescopes", lx, py + 250, 28, "Regular", GREY, a1, "center", shadow=6)
                k = 3 + 9 * L
                g = grating(P, k * bx, k * by)
                tile = torch.stack([g * 0.95, g * 0.85, g * 0.65], 0)
                img = paste(img, tile, rx - P / 2, py - P / 2, a1)
                frame_box(cv, rx - P / 2, py - P / 2, P, P, a1)
                cv.text("one ripple of the image", rx, py + 250, 28, "Regular", GREY, a1, "center", shadow=6)
            else:
                x = ramp(t, self.t_add, self.t_ft + 1.5, lambda q: q)
                N = max(1, int(round(2 ** (math.log2(self.n * self.n) * x ** 1.3))))
                rec, kidx = self.partial(N)
                ky, kx = divmod(int(kidx), self.n)
                ky = ky if ky < self.n / 2 else ky - self.n
                kx = kx if kx < self.n / 2 else kx - self.n
                phase = float(torch.angle(self.F.view(-1)[kidx]))
                g = grating(P, kx, -ky, phase)
                tile = torch.stack([g * 0.95, g * 0.85, g * 0.65], 0)
                img = paste(img, tile, lx - P / 2, py - P / 2, a1)
                frame_box(cv, lx - P / 2, py - P / 2, P, P, a1)
                pm = panel_image(rec.clamp(min=0).cpu().numpy(), P, norm=self.img_max * 1.05)
                img = paste(img, pm, rx - P / 2, py - P / 2, a1)
                frame_box(cv, rx - P / 2, py - P / 2, P, P, a1)
                cv.text("+", W / 2, py + 20, 70, "Light", WHITE, a1, "center")
                cv.text("ripple being added", lx, py + 250, 28, "Regular", GREY, a1, "center", shadow=6)
                cv.text(f"sum of {N:,} ripples", rx, py + 250, 28, "Medium", WHITE, a1, "center", shadow=6)
                a_ft = a1 * ramp(t, self.t_ft, self.t_ft + 0.6)
                cv.text("Fourier transform:  image  ⇄  ripples", W / 2, 120, 40, "Light", WHITE, a_ft, "center", shadow=8)
        # far apart -> fine, close together -> broad
        a2 = window(t, self.t_far - 0.3, self.t_turns + 0.4, 0.5, 0.6)
        if a2 > 0:
            for k, (Lb, kk, lab, sub) in enumerate(((1.0, 14, "far apart", "fine ripples"), (0.3, 4, "close together", "broad ripples"))):
                xc = W * (0.28 + 0.44 * k)
                half = 150 * Lb
                for sgn in (-1, 1):
                    dish_icon(cv, xc + sgn * half, py - 222, 36, WHITE, a2, angle=-90)
                cv.polyline([(xc - half, py - 230), (xc + half, py - 230)], AMBER, 2, a2, dash=(8, 6))
                cv.text(lab, xc, py - 280, 30, "Medium", WHITE, a2, "center", shadow=6)
                g = grating(360, kk, 0.0)
                img = paste(img, torch.stack([g * 0.95, g * 0.85, g * 0.65], 0), xc - 180, py - 150, a2)
                frame_box(cv, xc - 180, py - 150, 360, 360, a2)
                cv.text(sub, xc, py + 260, 28, "Regular", GREY, a2, "center", shadow=6)
        # the Earth turns: baselines sweep arcs through the Fourier plane
        a3 = ramp(t, self.t_turns, self.t_turns + 0.8)
        if a3 > 0:
            hours = 24.0 * ramp(t, self.t_turns + 0.3, self.duration - 0.8, lambda q: q)
            lon0 = sub_source_lon(hours)
            gx, gy, Rg = W * 0.27, H * 0.52, 300
            ge, gc = self.globe.render(M87_DEC_DEG, lon0, Rg, gx, gy, grid=0.06)
            img = img * (1 - gc * a3) + ge * a3
            u, v, tt, ii, jj = self.uv
            # stations and current baselines (stations that see M87 lie on this hemisphere)
            pos = []
            for (la, lo) in self.stations:
                pos.append(self.globe.project(la, lo, M87_DEC_DEG, lon0, Rg, gx, gy))
            sin10 = math.sin(math.radians(10))
            for a_i in range(len(pos)):
                for b_i in range(a_i + 1, len(pos)):
                    if pos[a_i][2] > sin10 and pos[b_i][2] > sin10:
                        col = self.pair_col.get((a_i, b_i), CYAN)
                        cv.polyline([pos[a_i][:2], pos[b_i][:2]], col, 2, 0.85 * a3)
            for (x, y, z) in pos:
                if z > 0:
                    cv.circle(x, y, 5, fill=WHITE if z > sin10 else GREY, alpha=a3 * smooth(z / 0.1), glow=4)
            cv.text("Earth, seen from M87", gx, gy + Rg + 60, 26, "Regular", GREY, a3, "center", shadow=6)
            cv.text(f"{hours:4.1f} h", gx, gy - Rg - 30, 30, "Medium", WHITE, a3, "center", shadow=6)
            # Fourier (u, v) plane
            ux, uy, US = W * 0.71, H * 0.52, 330
            gl = 9.5e9
            cv.rect(ux - US, uy - US, 2 * US, 2 * US, fill=(0.0, 0.0, 0.0), alpha=0.35 * a3)
            frame_box(cv, ux - US, uy - US, 2 * US, 2 * US, a3)
            cv.polyline([(ux - US, uy), (ux + US, uy)], GREY, 1, 0.4 * a3)
            cv.polyline([(ux, uy - US), (ux, uy + US)], GREY, 1, 0.4 * a3)
            cv.text("u", ux - US + 14, uy - 10, 22, "Regular", GREY, a3, shadow=4)
            cv.text("v", ux + 10, uy - US + 26, 22, "Regular", GREY, a3, shadow=4)
            m = tt <= hours
            for k in np.nonzero(m)[0]:
                col = self.pair_col.get((int(ii[k]), int(jj[k])), CYAN)
                for sgn in (1, -1):
                    X = ux - sgn * u[k] / gl * US  # u increases to the left (east), as on the sky
                    Y = uy - sgn * v[k] / gl * US
                    cv.circle(X, Y, 2.6, fill=col, alpha=0.9 * a3)
            cv.text("the Fourier plane: every point is one ripple", ux, uy - US - 30, 26, "Regular", WHITE, a3, "center", shadow=6)
        return over(img, cv.to_tensor())


# ============================================================================= 10. reconstruction


class ReconShot(Shot):
    name = "recon"

    def __init__(self):
        self.cues, self.duration = layout_cues(["rec1", "rec2", "rec3"], lead=0.6, gaps=[0.9, 1.6], tail=2.6)
        c1, c2, c3 = self.cues
        self.t_empty = c1.word("empty") - 0.4
        self.t_dirty = c1.word("Adding") - 0.3
        self.t_neg = c2.word("negative") - 0.6
        self.t_smooth = c2.word("smooth") - 0.6
        self.t_search = c2.word("search") - 0.3
        self.t_ring = c3.word("ring")

    def setup(self):
        d = sim()
        self.d = d
        # Fourier amplitude of the true image on a +-10 G-lambda grid (direct DFT, separable)
        from src.vlbi.eht import UAS
        img = torch.tensor(d["truth"], dtype=torch.float64, device=DEV)
        n = img.shape[0]
        pix = float(d["fov"]) / n * UAS
        c = (torch.arange(n, device=DEV, dtype=torch.float64) - (n - 1) / 2) * pix
        G = 256
        uu = torch.linspace(-10e9, 10e9, G, device=DEV, dtype=torch.float64)
        Eu = torch.exp(-2j * math.pi * uu[:, None] * (-c)[None, :])  # x = -c (east left)
        Ev = torch.exp(-2j * math.pi * uu[:, None] * (-c)[None, :])
        Fm = (Ev @ img.to(torch.complex128) @ Eu.T).abs()  # rows: v (top = +v), cols: u
        Fm = torch.flip(Fm, [0, 1])  # display: u to the left, v up
        self.amp = torch.log10(Fm / Fm.max() + 1e-4).float()
        self.amp = ((self.amp + 4) / 4).clamp(0, 1)
        self.snaps = d["snaps"]

    def frame(self, t):
        img = vignette_bg()
        cv = Canvas()
        d = self.d
        US = 300
        ux, uy = W * 0.29, H * 0.50
        a1 = 1 - ramp(t, self.t_neg - 0.6, self.t_neg + 0.2)
        # Fourier plane: full spectrum of the truth, then only what we sampled
        if a1 > 0:
            heat = icefire(self.amp ** 2.6)
            heat = Fn.interpolate(heat[None], size=(2 * US, 2 * US), mode="bilinear", align_corners=False)[0]
            dim = 1 - 0.85 * ramp(t, self.t_empty, self.t_empty + 1.2)
            img = paste(img, heat * dim, ux - US, uy - US, a1)
            frame_box(cv, ux - US, uy - US, 2 * US, 2 * US, a1)
            gl = 10e9
            u, v = d["u"], d["v"]
            a_pts = a1 * ramp(t, 0.3, 1.0)
            for k in range(0, len(u)):
                for sgn in (1, -1):
                    cv.circle(ux - sgn * u[k] / gl * US, uy - sgn * v[k] / gl * US, 2.2, fill=WHITE, alpha=0.85 * a_pts)
            cv.text("Fourier plane of the true image", ux, uy - US - 30, 26, "Regular", WHITE, a1 * (1 - ramp(t, self.t_empty, self.t_empty + 0.8)), "center", shadow=6)
            a_e = a1 * ramp(t, self.t_empty, self.t_empty + 0.8)
            cv.text("what we actually measured", ux, uy - US - 30, 26, "Regular", WHITE, a_e, "center", shadow=6)
            fill = 100 * len(u) * 2 * math.pi * 1.5 ** 2 / (2 * US) ** 2
            cv.text("most of the plane: empty", ux, uy + US + 50, 28, "Medium", CYAN, a_e, "center", shadow=6)
        rx, ry, P = W * 0.71, H * 0.50, 560
        # dirty image + beam
        a_d = window(t, self.t_dirty, self.t_search + 1.2, 0.6, 1.0)
        if a_d > 0:
            pm = panel_image(np.clip(d["dirty"], 0, None), P, gamma=1.4)
            img = paste(img, pm, rx - P / 2, ry - P / 2, a_d)
            frame_box(cv, rx - P / 2, ry - P / 2, P, P, a_d)
            cv.text("dirty image", rx, ry - P / 2 - 30, 30, "Medium", WHITE, a_d, "center", shadow=6)
            bm = panel_image(np.clip(d["beam"], 0, None), 170, cmap=icefire, gamma=0.6)
            img = paste(img, bm, rx + P / 2 - 180, ry + P / 2 - 180, a_d * ramp(t, self.t_dirty + 1.5, self.t_dirty + 2.2))
            cv.text("pattern of the gaps", rx + P / 2 - 95, ry + P / 2 + 30, 20, "Regular", GREY, a_d * ramp(t, self.t_dirty + 1.5, self.t_dirty + 2.2), "center", shadow=6)
        # what we know about real images
        a_k = ramp(t, self.t_neg, self.t_neg + 0.6)
        if a_k > 0:
            kx, ky = W * 0.11, H * 0.42
            cv.text("what we know:", kx, ky - 60, 30, "Light", GREY, a_k, shadow=6)
            for k, (tt, txt) in enumerate(((self.t_neg, "brightness is never negative"), (self.t_smooth, "the sky is mostly smooth"))):
                a_ = ramp(t, tt, tt + 0.6)
                cv.text("✓", kx, ky + k * 56, 32, "Medium", AMBER, a_, shadow=6)
                cv.text(txt, kx + 44, ky + k * 56, 30, "Regular", WHITE, a_, shadow=6)
            a_s = ramp(t, self.t_search, self.t_search + 0.6)
            cv.text("search: the simplest image", kx, ky + 160, 30, "Regular", WHITE, a_s, shadow=6)
            cv.text("that fits every measurement", kx, ky + 198, 30, "Regular", WHITE, a_s, shadow=6)
            cv.text("(closure phases & amplitudes)", kx, ky + 236, 22, "Regular", GREY, a_s, shadow=6)
        # RML iterations
        a_r = ramp(t, self.t_search, self.t_search + 0.8)
        if a_r > 0:
            x = ramp(t, self.t_search, self.t_ring + 0.4, lambda q: q)
            ns = len(self.snaps)
            kf = (ns - 1) * x ** 2.2
            k0 = int(kf)
            k1 = min(k0 + 1, ns - 1)
            f = kf - k0
            snap = self.snaps[k0] * (1 - f) + self.snaps[k1] * f
            pm = panel_image(snap, P, gamma=0.85)
            glow = ramp(t, self.t_ring - 0.2, self.t_ring + 0.8)
            img = paste(img, pm * (1 + 0.15 * glow), rx - P / 2, ry - P / 2, a_r)
            frame_box(cv, rx - P / 2, ry - P / 2, P, P, a_r)
            cv.text(f"iteration {int(kf):4d}", rx, ry + P / 2 + 44, 26, "Regular", GREY, a_r, "center", shadow=6)
            cv.text("reconstruction", rx, ry - P / 2 - 30, 30, "Medium", WHITE, a_r, "center", shadow=6)
        return over(img, cv.to_tensor())


# ============================================================================= 11. finale


class FinaleShot(Shot):
    name = "finale"

    def __init__(self):
        self.cues, self.duration = layout_cues(["end1", "end2"], lead=0.8, gap=1.8, tail=4.2)
        c1, c2 = self.cues
        self.t_ring = c1.word("ring")
        self.t_einstein = c1.word("Just") - 0.3
        self.t_hero = c2.start - 1.2

    def setup(self):
        d = sim()
        self.rec = d["rec"]
        self.tb = d["truth_blur"]
        hi = torch.tensor(d["truth_hi"], device=DEV)[None, None]
        self.hi = Fn.avg_pool2d(hi, 2)[0, 0].cpu().numpy()
        # soft circular field stop: hides the corners of the rotated simulation frame
        for name in ("hi", "tb", "rec"):
            a = getattr(self, name)
            n = a.shape[0]
            c = (np.arange(n) - (n - 1) / 2) / (n / 2)
            rr = np.sqrt(c[None, :] ** 2 + c[:, None] ** 2)
            setattr(self, name, a * np.clip((0.98 - rr) / 0.12, 0, 1) ** 2)

    def frame(self, t):
        img = vignette_bg()
        cv = Canvas()
        P = 460
        cy = H * 0.47
        grow = ramp(t, self.t_einstein, self.t_einstein + 2.0, ease_in_out)
        a_side = 1 - grow
        a_all = 1 - ramp(t, self.t_hero, self.t_hero + 1.4)
        xs = [W * 0.2, W * 0.5, W * 0.8]
        labels = [("our reconstruction", "from simulated EHT data"), ("simulation, blurred", "to EHT resolution"),
                  ("simulation, full detail", "photon rings visible")]
        ims = [panel_image(self.rec, P, gamma=0.85), panel_image(self.tb, P, gamma=0.85), panel_image(self.hi, P, gamma=0.55)]
        for k in range(3):
            if k == 2:
                Pk = int(lerp(P, 860, grow))
                xk = lerp(xs[2], W * 0.5, grow)
                tile = panel_image(self.hi, Pk, gamma=0.55)
                a = a_all * ramp(t, 0.2 + 0.4 * k, 0.9 + 0.4 * k)
                img = paste(img, tile, xk - Pk / 2, H * 0.5 - Pk / 2 + (cy - H * 0.5) * (1 - grow), a)
                if grow < 0.5:
                    frame_box(cv, xk - Pk / 2, cy - Pk / 2, Pk, Pk, a * (1 - 2 * grow))
            else:
                a = a_all * a_side * ramp(t, 0.2 + 0.4 * k, 0.9 + 0.4 * k)
                img = paste(img, ims[k], xs[k] - P / 2, cy - P / 2, a)
                frame_box(cv, xs[k] - P / 2, cy - P / 2, P, P, a)
            al = a_all * a_side * ramp(t, 0.2 + 0.4 * k, 0.9 + 0.4 * k)
            cv.text(labels[k][0], xs[k], cy + P / 2 + 50, 28, "Medium", WHITE, al, "center", shadow=6)
            cv.text(labels[k][1], xs[k], cy + P / 2 + 84, 22, "Regular", GREY, al, "center", shadow=6)
        # 42 uas scale on the reconstruction (160 uas field of view)
        a_s = a_all * a_side * ramp(t, self.t_ring - 0.3, self.t_ring + 0.5)
        if a_s > 0:
            L = P * 42 / 160
            x0 = xs[0] - L / 2
            yb = cy + P / 2 - 40
            cv.polyline([(x0, yb - 10), (x0, yb), (x0 + L, yb), (x0 + L, yb - 10)], WHITE, 2.5, a_s)
            cv.text("42 µas", xs[0], yb - 18, 26, "Medium", WHITE, a_s, "center", shadow=6)
        a_e = a_all * ramp(t, self.t_einstein + 1.2, self.t_einstein + 2.2)
        if a_e > 0:
            cv.text("what general relativity predicts", W / 2, H - 50, 30, "Light", WHITE, a_e, "center", shadow=8)
        # back to the black hole itself
        a_h = ramp(t, self.t_hero, self.t_hero + 1.6)
        if a_h > 0:
            from .shots_bh import HERO_LOOK, renderer
            from src.kerr.render import DiskStyle, Look
            from src.kerr.tracer import Camera
            R = renderer()
            R.set_spin(0.9)
            x = (t - self.t_hero) / (self.duration - self.t_hero)
            cam = Camera(r=loglerp(58, 120, ease_in_out(x)), theta=math.radians(lerp(83.5, 85.5, x)),
                         phi=math.radians(lerp(10, 22, x)), fov=34)
            hero, _ = R.render(cam, W, H, ss=3, style=DiskStyle(T_max=7000, t=4000 + 14 * t), look=Look(**HERO_LOOK),
                               seed=int(t * FPS))
            fade = 1 - ramp(t, self.duration - 2.6, self.duration - 0.2)
            img = img * (1 - a_h) + hero * a_h * fade
        return over(img, cv.to_tensor())


# ============================================================================= 12. credits


class CreditsShot(Shot):
    name = "credits"
    LINES = [
        ("HOW TO PHOTOGRAPH A BLACK HOLE", 46, "DisplayLight", WHITE),
        ("", 20, "Regular", GREY),
        ("Every image in this film was computed. None are photographs.", 28, "Regular", WHITE),
        ("", 26, "Regular", GREY),
        ("Black holes: Kerr null geodesics traced on the GPU (adaptive RK45, FP32),", 24, "Regular", GREY),
        ("validated against √27 M and Bardeen's 1973 shadow to under one pixel", 24, "Regular", GREY),
        ("Disk: Novikov–Thorne flux · Doppler and gravitational redshift · blackbody color", 24, "Regular", GREY),
        ("M87*-like image: optically thin emission (Gralla, Lupsasca & Marrone 2020 profiles)", 24, "Regular", GREY),
        ("Telescope: simulated 2017 EHT array, thermal noise, atmospheric phases,", 24, "Regular", GREY),
        ("closure-phase & closure-amplitude regularized imaging", 24, "Regular", GREY),
        ("Narration: Kokoro-82M speech synthesis · Music: synthesized from scratch", 24, "Regular", GREY),
        ("Coastlines: Natural Earth (public domain)", 24, "Regular", GREY),
        ("", 26, "Regular", GREY),
        ("The real images of M87* and Sgr A*: Event Horizon Telescope Collaboration, 2019 & 2022", 24, "Regular", WHITE),
        ("", 26, "Regular", GREY),
        ("Written, coded and rendered by Claude (Anthropic) · October 2026", 26, "Medium", AMBER),
    ]

    def __init__(self):
        self.cues, self.duration = [], 17.0

    def frame(self, t):
        img = vignette_bg()
        cv = Canvas()
        y = H * 0.5 - sum(int(s * 1.55) for _, s, _, _ in self.LINES) / 2 + 30
        for k, (txt, size, wt, col) in enumerate(self.LINES):
            a = window(t, 0.6 + 0.18 * k, self.duration - 0.3, 0.9, 1.6)
            if txt:
                cv.text(txt, W / 2, y, size, wt, col, a, "center", tracking=0.04 if k == 0 else 0.0)
            y += int(size * 1.55)
        return over(img, cv.to_tensor())


_TAIL.extend([(ScaleShot, 1.0), (EHTShot, 1.0), (FourierShot, 1.0), (ReconShot, 1.0), (FinaleShot, 1.0), (CreditsShot, 0.0)])
