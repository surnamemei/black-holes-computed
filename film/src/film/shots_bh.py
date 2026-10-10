"""Shots rendered with the Kerr ray tracer: cold open, lensing, disk, Doppler, spin."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from src.kerr.render import DiskStyle, Look, Renderer
from src.kerr.tracer import ST_HORIZON, Camera

from .core import (AMBER, BLUE, CYAN, DEV, FPS, GREY, H, RED, W, WHITE, Canvas, Shot, ease_in_out, layout_cues, lerp,
                   loglerp, over, ramp, smooth, smoother, window)

SS = 3  # supersampling (3x3 rays per pixel)
_R = None


def renderer():
    global _R
    if _R is None:
        _R = Renderer(a=0.0)
    return _R


HERO_LOOK = dict(exposure=3.0, white=4.0, saturation=1.45, bloom=0.035)


def gal_center_dir():
    """World direction of the procedural galaxy's bulge (to frame the band behind the hole)."""
    R = renderer().sky.R_gal.cpu().numpy()
    return R.T @ np.array([1.0, 0.0, 0.0])


def dir_to_angles(d):
    d = d / np.linalg.norm(d)
    return math.acos(np.clip(d[2], -1, 1)), math.atan2(d[1], d[0])


def label(cv: Canvas, x, y, tx, ty, text, alpha, color=WHITE, size=30, sub=None):
    """Leader line from (x, y) on the subject to a text anchor (tx, ty) (clamped on screen)."""
    if alpha <= 0:
        return
    tx = min(max(tx, 330), W - 330)
    ty = min(max(ty, 80), H - 110)
    cv.circle(x, y, 5, fill=color, alpha=alpha, glow=4)
    mx = tx - 14 if tx > x else tx + 14
    cv.polyline([(x, y), (mx, ty), (tx, ty)], color, 1.6, alpha * 0.9)
    al = "left" if tx > x else "right"
    tx2 = tx + 10 if tx > x else tx - 10
    cv.text(text, tx2, ty + 10, size, "Medium", color, alpha, align=al, shadow=6)
    if sub:
        cv.text(sub, tx2, ty + 10 + size * 1.15, size * 0.72, "Regular", GREY, alpha, align=al, shadow=6)


def mask_edge(m):
    """Soft outline of a [0,1] mask (1,H,W) -> (1,H,W)."""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=DEV).view(1, 1, 3, 3)
    e = Fn.conv2d(m[None], k, padding=1)[0].abs()
    return Fn.avg_pool2d(e[None], 3, stride=1, padding=1)[0].clamp(0, 1)


def centroid(m):
    tot = m.sum()
    if tot < 1:
        return None
    ys = torch.arange(m.shape[-2], device=DEV).float()
    xs = torch.arange(m.shape[-1], device=DEV).float()
    return float((m.sum(-2) * xs).sum() / tot), float((m.sum(-1) * ys).sum() / tot)


def mask_point(m, target, thresh=0.5):
    """The mask pixel (value > thresh) nearest to a target screen point."""
    mm = m[0] > thresh
    if mm.sum() < 5:
        return None
    ys, xs = torch.nonzero(mm, as_tuple=True)
    d = (xs.float() - target[0]) ** 2 + (ys.float() - target[1]) ** 2
    i = torch.argmin(d)
    return float(xs[i]), float(ys[i])


def shadow_band(res, width=7):
    """Pixels just outside the shadow edge (where the photon ring lives), at output resolution."""
    cap = Fn.avg_pool2d((res.status == ST_HORIZON).float()[None, None], SS)[0]
    inside = (cap > 0.5).float()
    dil = Fn.max_pool2d(inside[None], 2 * width + 1, stride=1, padding=width)[0]
    return (dil - inside).clamp(0, 1), cap


def extreme_point(m, side="top"):
    """A point on the mask: its topmost/bottommost/leftmost/rightmost well-covered pixel."""
    mm = m[0] > 0.5
    if mm.sum() < 5:
        return None
    ys, xs = torch.nonzero(mm, as_tuple=True)
    if side == "top":
        i = torch.argmin(ys)
    elif side == "bottom":
        i = torch.argmax(ys)
    elif side == "left":
        i = torch.argmin(xs)
    else:
        i = torch.argmax(xs)
    return float(xs[i]), float(ys[i])


# ============================================================================= 1. cold open


class OpenShot(Shot):
    name = "open"

    def __init__(self):
        self.cues, self.duration = layout_cues(["open1", "open2"], lead=3.5, gap=1.0, tail=7.0)
        self.title_t0 = self.cues[-1].end + 0.6

    def setup(self):
        renderer()

    def frame(self, t):
        R = renderer()
        R.set_spin(0.9)
        x = t / self.duration
        cam = Camera(r=loglerp(170, 64, ease_in_out(min(1, x * 1.15))), theta=math.radians(lerp(87.5, 82.5, ease_in_out(x))),
                     phi=math.radians(lerp(-6, 4, x)), fov=34, forward=(-1.0, lerp(0.015, 0.0, x), lerp(0.03, -0.004, x)))
        style = DiskStyle(T_max=7000, t=300 + 14 * t)
        look = Look(**HERO_LOOK)
        img, _ = R.render(cam, W, H, ss=SS, style=style, look=look, seed=int(t * FPS))
        img = img * ramp(t, 0.0, 3.2, smooth)
        # title over a soft darkening at the bottom
        a = window(t, self.title_t0, self.duration + 0.5, 1.6, 1.2)
        if a > 0:
            yy = torch.linspace(0, 1, H, device=DEV).view(1, H, 1)
            shade = 1 - 0.55 * a * smooth(((yy - 0.55) / 0.45).clamp(0, 1))
            img = img * shade
            cv = Canvas()
            ty = H * 0.80
            trk = lerp(0.06, 0.03, ease_in_out((t - self.title_t0) / 6))
            cv.text("HOW TO PHOTOGRAPH A BLACK HOLE", W / 2, ty, 62, "DisplayLight", WHITE, a, "center", tracking=trk, shadow=10)
            cv.text("every frame computed from general relativity", W / 2, ty + 54, 25, "Light", GREY,
                    a * ramp(t, self.title_t0 + 1.0, self.title_t0 + 2.4), "center", tracking=0.12, shadow=6)
            img = over(img, cv.to_tensor())
        return img


# ============================================================================= 2. lensing


class LensingShot(Shot):
    name = "lensing"
    R_FINAL = 40.0

    def __init__(self):
        self.cues, self.duration = layout_cues(["lens1", "lens2"], lead=1.0, gap=1.0, tail=2.2)
        c2 = self.cues[1]
        self.t_add = c2.word("add")
        self.t_ramp1 = self.t_add + 3.6
        self.t_align = c2.end - 0.5  # "...folded into rings"

    def setup(self):
        renderer()
        # look along the band ~55 deg from the bulge, so it crosses the frame as a defined stripe
        R = renderer().sky.R_gal.cpu().numpy()
        l, b = math.radians(55), math.radians(4)
        g = R.T @ np.array([math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b)])
        cam_dir = -g
        self.th0, self.ph0 = dir_to_angles(cam_dir)
        # a background galaxy exactly behind the hole at t_align: it becomes an Einstein ring
        if not getattr(renderer().sky, "_lens_galaxy", False):
            cam = self.camera(self.t_align)
            th, ph = cam.theta, cam.phi
            behind = -np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            renderer().sky.add_galaxy(behind, size_deg=6.0, tilt_deg=40, pa_deg=25, brightness=0.28)
            renderer().sky._lens_galaxy = True

    def camera(self, t):
        r = self.R_FINAL if t >= self.t_ramp1 else loglerp(2.0e6, self.R_FINAL, ease_in_out((t - self.t_add) / (self.t_ramp1 - self.t_add))) if t > self.t_add else 2.0e6
        orbit = math.radians(34) * ease_in_out((t - self.t_add - 1.5) / (self.duration - self.t_add - 1.5))
        pan = math.radians(14) * (1 - ease_in_out(t / (self.t_add + 0.5)))
        return Camera(r=r, theta=self.th0 - 0.12 * orbit, phi=self.ph0 + orbit, fov=70,
                      forward=(-math.cos(pan), 0.03 * (1 - smooth(t / self.t_add)), math.sin(pan)))

    def frame(self, t):
        R = renderer()
        R.set_spin(0.0)
        cam = self.camera(t)
        look = Look(exposure=4.0, white=3.0, saturation=1.2, bloom=0.03, sky_gain=1.0)
        if t <= self.t_add:
            img = R.render_flat(cam, W, H, ss=2, look=look, seed=int(t * FPS))
        else:
            img, _ = R.render(cam, W, H, ss=SS, look=look, disk_on=False, seed=int(t * FPS))
        return img


# ============================================================================= 4. disk


def disk_camera(t_abs):
    """Shared camera path for the disk and Doppler shots (t_abs: seconds since the disk shot began)."""
    return Camera(r=lerp(78, 66, smooth(t_abs / 60)), theta=math.radians(lerp(82.5, 81.0, smooth(t_abs / 60))),
                  phi=math.radians(lerp(0, 14, t_abs / 60)), fov=34, forward=(-1.0, 0.0, 0.0))


DISK_T = 5200.0
DISK_LOOK = dict(exposure=1.7, white=4.0, saturation=1.4, bloom=0.035)


class DiskShot(Shot):
    name = "disk"

    def __init__(self):
        self.cues, self.duration = layout_cues(["disk1", "disk2", "disk3"], lead=1.2, gap=0.8, tail=1.0)
        c1, c2, c3 = self.cues
        self.t_disk = c1.word("disk") - 1.2
        self.t_far = c2.word("far")
        self.t_top = c2.word("top")
        self.t_under = c2.word("underneath")
        self.t_ring = c3.word("ring")
        self.t_loop = c3.word("looped")

    def setup(self):
        renderer()

    def frame(self, t):
        R = renderer()
        R.set_spin(0.0)
        cam = disk_camera(t)
        fade = ramp(t, self.t_disk, self.t_disk + 3.5, smoother)
        style = DiskStyle(T_max=DISK_T, doppler="none", t=900 + 14 * t, brightness=fade, opacity=fade)
        look = Look(**DISK_LOOK)
        hdr, res = R.render(cam, W, H, ss=SS, style=style, look=look, return_hdr=True)
        cv = Canvas()
        hits = None
        if t > self.t_far - 0.5:
            hits = R.first_hit(res, style)
            far = torch.cos(hits["phi"] - cam.phi) < 0
            hit = hits["k"] >= 0
            # photon sgn < 0: the backward ray comes down onto the top face; > 0: up onto the bottom face
            top = (hits["sgn"] < 0) & far & (hits["k"] == 0)
            bot = (hits["sgn"] > 0) & far & hit
            pool = lambda m: Fn.avg_pool2d(m.float()[None, None], SS)[0]
            band, _ = shadow_band(res)
            mt, mb = pool(top), pool(bot) * (1 - band)
            mr = band * (pool(hit) > 0.3).float()
            a_top = window(t, self.t_far - 0.2, self.t_ring - 0.4, 0.6, 0.8)
            a_bot = window(t, self.t_under - 0.3, self.t_ring - 0.4, 0.6, 0.8)
            a_ring = window(t, self.t_ring - 0.3, self.duration + 1.0, 0.5, 0.8)
            pulse = lambda t0: 0.5 + 0.5 * math.cos(2 * math.pi * max(0.0, t - t0) / 1.6)
            boost = 1 + 0.55 * (a_top * pulse(self.t_far) * mt + a_bot * pulse(self.t_under) * mb + 0.9 * a_ring * pulse(self.t_ring) * mr)
            hdr = hdr * boost
            img = R.post(hdr, look, seed=int(t * FPS))
            pt = mask_point(mt, (W * 0.62, 0))
            if pt and a_top > 0:
                label(cv, pt[0], pt[1] + 12, pt[0] + 300, pt[1] - 30, "far side of the disk", a_top, AMBER, 30,
                      "bent up over the top")
            pb = mask_point(mb, (W * 0.40, H))
            if pb and a_bot > 0:
                label(cv, pb[0], pb[1] - 12, pb[0] - 300, pb[1] + 40, "its underside", a_bot, AMBER, 30,
                      "bent down underneath")
            pr = mask_point(mr, (W * 0.25, H * 0.15))
            if pr and a_ring > 0:
                label(cv, pr[0], pr[1], pr[0] - 280, pr[1] - 120, "photon ring", a_ring, CYAN, 30,
                      "light that looped around the hole")
        else:
            img = R.post(hdr, look, seed=int(t * FPS))
        return over(img, cv.to_tensor())


# ============================================================================= 5. Doppler


def project(cam: Camera, P):
    """Flat-space pinhole projection of world points P (N,3) for a camera looking at the hole
    (good enough for schematic arrows placed outside the strongly lensed region)."""
    th, ph = cam.theta, cam.phi
    C = cam.r * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    er = C / cam.r
    et = np.array([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th)])
    ep = np.array([-math.sin(ph), math.cos(ph), 0.0])
    f = -er
    u = -et
    rgt = np.cross(f, u)
    v = P - C
    z = v @ f
    tx = math.tan(math.radians(cam.fov) / 2)
    x = (v @ rgt) / z / tx
    y = (v @ u) / z / (tx * H / W)
    return np.stack([W / 2 + x * W / 2, H / 2 - y * H / 2], -1)


def orbit_arrow(cv, cam, r, phi0, phi1, color, alpha, width=4.0, n=40):
    """Arrow along a circular equatorial orbit from azimuth phi0 to phi1 (radians, world frame)."""
    ph = np.linspace(phi0, phi1, n)
    P = np.stack([r * np.cos(ph), r * np.sin(ph), np.zeros_like(ph)], -1)
    pts = project(cam, P)
    cv.polyline([tuple(p) for p in pts], color, width, alpha, glow=2, glow_alpha=0.2)
    (x0, y0), (x1, y1) = pts[-3], pts[-1]
    ang = math.atan2(y1 - y0, x1 - x0)
    for s_ in (-1, 1):
        a_ = ang + math.pi + s_ * 0.5
        cv.polyline([(x1, y1), (x1 + 18 * math.cos(a_), y1 + 18 * math.sin(a_))], color, width, alpha)
    return pts


class DopplerShot(Shot):
    name = "doppler"

    def __init__(self, t_offset):
        self.t_offset = t_offset  # continue the disk camera seamlessly
        self.cues, self.duration = layout_cues(["dopp1", "dopp2", "dopp3"], lead=0.4, gap=0.8, tail=1.8)
        c1, c2, c3 = self.cues
        self.t_speed = c1.word("half")
        self.t_toward = c2.word("rushing")
        self.t_recede = c2.word("receding")
        self.t_dim = c2.word("dimmer") + 0.6
        self.t_grav = c2.word("Climbing")

    def frame(self, t):
        R = renderer()
        R.set_spin(0.0)
        cam = disk_camera(t + self.t_offset)
        mix = ramp(t, self.t_toward - 0.3, self.t_dim, ease_in_out)
        style = DiskStyle(T_max=DISK_T, doppler="full", doppler_mix=mix, t=900 + 14 * (t + self.t_offset))
        look = Look(**DISK_LOOK)
        img, res = R.render(cam, W, H, ss=SS, style=style, look=look, seed=int(t * FPS))
        cv = Canvas()
        # orbital-speed callout
        a_v = window(t, self.t_speed - 0.2, self.t_toward - 0.2, 0.5, 0.6)
        cx, cy = W * 0.5, H * 0.52
        if a_v > 0:
            cv.text("v ≈ 0.5 c", cx, cy + 165, 34, "Medium", WHITE, a_v, "center", shadow=8)
            cv.text("orbital speed at the inner edge", cx, cy + 200, 22, "Regular", GREY, a_v, "center", shadow=6)
        a1 = window(t, self.t_toward - 0.2, self.duration + 1, 0.6, 0.8)
        a2 = window(t, self.t_recede - 0.2, self.duration + 1, 0.6, 0.8)
        # gas orbits in +phi: the left limb (phi_cam - 90 deg) comes towards us, the right limb recedes
        pc = cam.phi
        if a1 > 0:
            pts = orbit_arrow(cv, cam, 17.0, pc - math.radians(135), pc - math.radians(98), BLUE, a1, width=3.0)
            cv.text("approaching", pts[-1][0] + 10, pts[-1][1] + 70, 30, "Medium", BLUE, a1, "center", shadow=8, backdrop=0.55)
            cv.text("bluer, brighter", pts[-1][0] + 10, pts[-1][1] + 104, 22, "Regular", WHITE, a1, "center", shadow=6, backdrop=0.55)
        if a2 > 0:
            pts = orbit_arrow(cv, cam, 17.0, pc + math.radians(82), pc + math.radians(125), RED, a2, width=3.0)
            cv.text("receding", pts[0][0] - 10, pts[0][1] + 70, 30, "Medium", RED, a2, "center", shadow=8, backdrop=0.55)
            cv.text("redder, dimmer", pts[0][0] - 10, pts[0][1] + 104, 22, "Regular", WHITE, a2, "center", shadow=6, backdrop=0.55)
        a3 = window(t, self.t_grav - 0.2, self.duration - 0.2, 0.6, 0.8)
        if a3 > 0:
            cv.text("+ gravitational redshift near the hole", W / 2, H * 0.93, 26, "Regular", WHITE, a3, "center", shadow=8)
        return over(img, cv.to_tensor())


# ============================================================================= 6. spin


class SpinShot(Shot):
    name = "spin"

    def __init__(self):
        self.cues, self.duration = layout_cues(["spin1", "spin2"], lead=0.8, gap=0.8, tail=2.6)
        c1, c2 = self.cues
        self.t0 = c1.word("spin") - 0.2
        self.t1 = c2.word("away") + 0.6
        self.t_with = c2.word("with")
        self.t_against = c2.word("against")
        self.t_flat = c2.word("flattened")

    def frame(self, t):
        R = renderer()
        a = 0.99 * ease_in_out((t - self.t0) / (self.t1 - self.t0))
        R.set_spin(a, flux_norm="schwarzschild")
        cam = Camera(r=58, theta=math.radians(84.0), phi=0.0, fov=33, forward=(-1.0, 0.0, 0.0))
        style = DiskStyle(T_max=DISK_T, doppler="full", t=2000 + 14 * t)
        look = Look(**DISK_LOOK)
        hdr, res = R.render(cam, W, H, ss=SS, style=style, look=look, return_hdr=True)
        img = R.post(hdr, look, seed=int(t * FPS))
        # shadow outline (rays that end on the horizon), drawn as a glowing contour
        cap = Fn.avg_pool2d((res.status == ST_HORIZON).float()[None, None], SS)[0]
        edge = mask_edge(cap)
        glow = Fn.avg_pool2d(edge[None], 9, stride=1, padding=4)[0]
        a_edge = ramp(t, self.t0 - 0.5, self.t0 + 0.5)
        col = torch.tensor(CYAN, device=DEV).view(3, 1, 1)
        img = img + a_edge * (1.0 * edge + 1.5 * glow).clamp(0, 1) * col * 0.8
        cv = Canvas()
        cv.text(f"spin  a = {a:.2f}", 90, 110, 38, "Medium", WHITE, ramp(t, 0.3, 1.0), shadow=8)
        cv.text("shadow edge", 90, 150, 24, "Regular", CYAN, a_edge, shadow=6)
        # where the edge is: leftmost/rightmost points of the shadow mask
        pl = extreme_point(cap, "left")
        prr = extreme_point(cap, "right")
        a_w = window(t, self.t_with - 0.2, self.duration + 1, 0.5, 0.6)
        a_a = window(t, self.t_against - 0.2, self.duration + 1, 0.5, 0.6)
        if pl and a_w > 0:
            label(cv, pl[0], pl[1], pl[0] - 260, pl[1] + 250, "with the spin", a_w, CYAN, 28, "orbits closer in")
        if prr and a_a > 0:
            label(cv, prr[0], prr[1], prr[0] + 240, prr[1] + 250, "against the spin", a_a, CYAN, 28, "pushed away")
        return over(img, cv.to_tensor())
