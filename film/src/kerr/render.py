"""Frame renderer: trace -> shade (disk + sky) -> resolve -> bloom -> tone map."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from .emission import BlackbodyLUT, Disk
from .sky import Sky, fbm3
from .tracer import MAX_CROSS, ST_ESCAPED, Camera, trace

# ----------------------------------------------------------------------------- disk texture


def make_disk_texture(nr=1024, nphi=4096, seed=3, device="cuda"):
    """Brightness modulation on (log r, phi), periodic in phi; values ~ [0.2, 1.8]."""
    u = torch.linspace(0, 1, nr, device=device)
    ph = torch.linspace(0, 2 * math.pi, nphi + 1, device=device)[:-1]
    U, P = torch.meshgrid(u, ph, indexing="ij")
    tex = torch.zeros(nr, nphi, device=device)
    # anisotropic fbm: fine structure in radius, long streaks in azimuth
    for k, (fr, fp, amp) in enumerate([(14, 1.5, 1.0), (40, 3.0, 0.6), (110, 6.0, 0.35), (260, 12.0, 0.2)]):
        p = torch.stack([U * fr, torch.cos(P) * fp, torch.sin(P) * fp], -1)
        tex += amp * (fbm3(p, octaves=3, seed=seed + 11 * k) - 0.5)
    tex = tex / tex.std()
    clumps = fbm3(torch.stack([U * 9, torch.cos(P) * 2.2, torch.sin(P) * 2.2], -1), octaves=4, seed=seed + 99)
    tex = torch.exp(0.55 * tex + 1.4 * (clumps - 0.5))
    tex = tex / tex.mean()
    return tex


@dataclass
class DiskStyle:
    T_max: float = 9000.0  # emitted temperature at the flux peak [K]
    r_out: float = 24.0  # where the visible disk fades out
    r_fade: float = 7.0  # width of the outer fade
    texture: float = 1.0  # 0 = smooth Novikov-Thorne disk, 1 = full texture
    doppler: str = "full"  # "full" | "nolos" (no line-of-sight Doppler) | "none" (g = 1)
    doppler_mix: float = 1.0  # g_eff = g ** mix: blends "none" (0) to "full" (1)
    opacity: float = 1.0  # opacity of the bright disk
    brightness: float = 1.0
    t: float = 0.0  # time in M (rotates the texture differentially)


@dataclass
class Look:
    exposure: float = 1.0
    bloom: float = 0.035
    sky_gain: float = 1.0
    vignette: float = 0.18
    grain: float = 0.006
    saturation: float = 1.0
    white: float = 6.0  # scene luminance that maps to display white


class Renderer:
    def __init__(self, a=0.9, sky: Sky | None = None, device="cuda"):
        self.a = a
        self.disk = Disk(a)
        self.bb = BlackbodyLUT(device=device)
        self.sky = sky if sky is not None else Sky(device=device)
        self.tex = make_disk_texture(device=device)
        self.tex_logr0, self.tex_logr1 = math.log(1.0), math.log(80.0)  # fixed: spin changes must not slide it
        # luminance reference: blackbody at 9000 K
        _, self.Y_ref = self.bb.lookup(torch.tensor([9000.0], device=device))

    def set_spin(self, a, flux_norm="self"):
        if abs(a - self.a) > 1e-9 or getattr(self, "_fnorm", "self") != flux_norm:
            self.a = a
            self._fnorm = flux_norm
            self.disk = Disk(a, flux_norm=flux_norm)

    # ------------------------------------------------------------------ shading
    def _tex_lookup(self, r, phi_c, shift):
        u = (torch.log(r) - self.tex_logr0) / (self.tex_logr1 - self.tex_logr0) * 2 - 1
        v = torch.remainder(phi_c + shift, 2 * math.pi) / math.pi - 1
        grid = torch.stack([v, u], -1).view(1, 1, -1, 2)
        s = Fn.grid_sample(self.tex[None, None], grid, mode="bilinear", padding_mode="border", align_corners=False)
        return s.view(-1)

    def disk_texture(self, r, phi, t, period=90.0):
        """Flow-map animation: two copies of the texture advected by Keplerian rotation, each
        reset every `period` (in M) and cross-faded, so the gas visibly orbits while the shear
        stays bounded (as turbulence keeps regenerating structure in a real disk)."""
        out = 0
        for k, off in enumerate((0.0, 0.5)):
            ph = (t / period + off)
            cyc = math.floor(ph)
            frac = ph - cyc
            w = 1 - abs(2 * frac - 1)
            age = frac * period
            shift = (cyc * 2.399 + k * 1.3) % (2 * math.pi)  # new random-looking offset each cycle
            out = out + w * self._tex_lookup(r, phi - self.disk.orbital_phase(r, age), shift)
        return out

    def shade_disk_crossing(self, r, phi, pr, b, E_cam, style: DiskStyle):
        F = self.disk.flux(r)
        fade = torch.clamp((style.r_out - r) / style.r_fade, 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        if style.texture > 0:
            m = self.disk_texture(r, phi, style.t)
            m = 1 + style.texture * (m - 1)
        else:
            m = torch.ones_like(r)
        T_emit = style.T_max * torch.clamp(F * m, min=0) ** 0.25
        if style.doppler == "full":
            g = self.disk.redshift(r, b, pr, E_cam)
            if style.doppler_mix != 1.0:
                g = g ** style.doppler_mix
        elif style.doppler == "nolos":
            g = E_cam / self_ut(self.disk, r)
        else:
            g = torch.ones_like(r)
        T_obs = g * T_emit
        rgb, Y = self.bb.lookup(T_obs)
        L = style.brightness * (Y / self.Y_ref)[:, None] * rgb
        # opacity follows the visible disk; inside the ISCO the flow is transparent here
        alpha = style.opacity * fade * (r >= self.disk.r_isco).float()
        return L * fade[:, None], alpha

    def render(self, cam: Camera, width, height, ss=3, style: DiskStyle | None = None, look: Look | None = None,
               jitter=False, disk_on=True, sky_on=True, return_hdr=False, seed=0):
        style = style or DiskStyle()
        look = look or Look()
        res = trace(cam, self.a, width, height, ss=ss, jitter=jitter)
        Hs, Ws = res.status.shape
        N = Hs * Ws
        st = res.status.view(-1)
        nc = res.ncross.view(-1)
        cross = res.cross.view(N, MAX_CROSS, 4)
        # drop crossings after any non-finite record (pathological rays near a pole)
        bad = ~torch.isfinite(cross).all(-1)
        first_bad = torch.where(bad.any(-1), bad.float().argmax(-1), torch.full_like(nc, MAX_CROSS).long())
        nc = torch.minimum(nc.long(), first_bad)
        b = res.b.view(-1)
        E_cam = res.E_cam.view(-1)
        col = torch.zeros(N, 3, device="cuda")
        trans = torch.ones(N, device="cuda")
        if disk_on:
            for k in range(MAX_CROSS):
                m = (nc > k) & (trans > 1e-4)
                if not m.any():
                    break
                idx = m.nonzero().squeeze(1)
                r = cross[idx, k, 0]
                L, al = self.shade_disk_crossing(r, cross[idx, k, 1], cross[idx, k, 2], b[idx], E_cam[idx], style)
                # front-to-back compositing of an emitting layer with opacity alpha
                tr = trans[idx]
                al = al.clamp(0, 1)
                col[idx] += (tr * al)[:, None] * L
                trans[idx] = tr * (1 - al)
        if sky_on:
            esc = (st == ST_ESCAPED)
            d = res.esc_dir
            # angular footprint of each sample from its neighbours in the supersampled grid
            dx = torch.zeros(Hs, Ws, device="cuda")
            dy = torch.zeros(Hs, Ws, device="cuda")
            e2 = esc.view(Hs, Ws)
            ddx = (d[:, 1:] - d[:, :-1]).norm(dim=-1)
            ddy = (d[1:, :] - d[:-1, :]).norm(dim=-1)
            okx = e2[:, 1:] & e2[:, :-1]
            oky = e2[1:, :] & e2[:-1, :]
            dx[:, :-1] = torch.where(okx, ddx, dx[:, :-1])
            dx[:, 1:] = torch.maximum(dx[:, 1:], torch.where(okx, ddx, torch.zeros_like(ddx)))
            dy[:-1, :] = torch.where(oky, ddy, dy[:-1, :])
            dy[1:, :] = torch.maximum(dy[1:, :], torch.where(oky, ddy, torch.zeros_like(ddy)))
            pix = math.radians(cam.fov) / Ws
            fp = torch.maximum(torch.maximum(dx, dy), torch.full_like(dx, pix)).view(-1)
            ie = esc.nonzero().squeeze(1)
            if ie.numel():
                skyc = self.sky.sample(d.view(-1, 3)[ie], fp[ie])
                # relativistic boost of the sky seen by the camera: I ~ g^4 (bolometric)
                gs = E_cam[ie]
                col[ie] += trans[ie, None] * look.sky_gain * skyc * (gs ** 4)[:, None]
        img = col.view(Hs, Ws, 3).permute(2, 0, 1)[None]
        if ss > 1:
            img = Fn.avg_pool2d(img, ss)
        img = img[0]  # (3, H, W) linear HDR
        if return_hdr:
            return img, res
        return self.post(img, look, seed=seed), res

    def render_flat(self, cam: Camera, width, height, ss=2, look: Look | None = None, return_hdr=False, seed=0):
        """No black hole: straight rays to the celestial sphere (same orientation conventions)."""
        from .tracer import camera_rays

        look = look or Look()
        init = camera_rays(cam, 0.0, width, height, ss)
        n = init["n_cam"]  # (H, W, 3) in (r, th, ph) components
        th, ph = cam.theta, cam.phi
        er = torch.tensor([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)], device="cuda")
        et = torch.tensor([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th)], device="cuda")
        ep = torch.tensor([-math.sin(ph), math.cos(ph), 0.0], device="cuda")
        d = n[..., 0:1] * er + n[..., 1:2] * et + n[..., 2:3] * ep
        Hs, Ws = d.shape[:2]
        fp = torch.full((Hs * Ws,), math.radians(cam.fov) / Ws, device="cuda")
        col = look.sky_gain * self.sky.sample(d.reshape(-1, 3), fp)
        img = col.view(Hs, Ws, 3).permute(2, 0, 1)[None]
        if ss > 1:
            img = Fn.avg_pool2d(img, ss)
        img = img[0]
        if return_hdr:
            return img
        return self.post(img, look, seed=seed)

    def first_hit(self, res, style: DiskStyle, alpha_min=0.5):
        """Per supersample: index of the first equatorial crossing that lands on the visible disk
        (-1 if none), with its radius, azimuth and the sign of the photon's p_theta there."""
        Hs, Ws = res.status.shape
        N = Hs * Ws
        cross = res.cross.view(N, MAX_CROSS, 4)
        nc = res.ncross.view(-1)
        k_hit = torch.full((N,), -1, dtype=torch.long, device="cuda")
        for k in range(MAX_CROSS):
            r = cross[:, k, 0]
            fade = torch.clamp((style.r_out - r) / style.r_fade, 0, 1)
            ok = (nc > k) & (k_hit < 0) & (r >= self.disk.r_isco) & (fade > alpha_min)
            k_hit = torch.where(ok, torch.full_like(k_hit, k), k_hit)
        kk = k_hit.clamp(min=0)
        sel = cross[torch.arange(N, device="cuda"), kk]
        return dict(k=k_hit.view(Hs, Ws), r=sel[:, 0].view(Hs, Ws), phi=sel[:, 1].view(Hs, Ws), sgn=sel[:, 3].view(Hs, Ws))

    # ------------------------------------------------------------------ post
    def post(self, img, look: Look, seed=0):
        img = img * look.exposure
        if look.bloom > 0:
            img = img + look.bloom * bloom(img)
        out = filmic(img, look.white)
        if look.saturation != 1.0:
            lum = (0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2])[None]
            out = lum + look.saturation * (out - lum)
        C, H, W = out.shape
        if look.vignette > 0:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=out.device), torch.linspace(-1, 1, W, device=out.device), indexing="ij")
            rr = (xx ** 2 * (W / H) ** 2 + yy ** 2) / (1 + (W / H) ** 2)
            out = out * (1 - look.vignette * rr ** 1.5)[None]
        out = srgb_oetf(out.clamp(0, 1))
        g = torch.Generator(device=out.device).manual_seed(seed)
        if look.grain > 0:
            out = out + look.grain * torch.randn(1, H, W, device=out.device, generator=g)
        out = out + (torch.rand(3, H, W, device=out.device, generator=g) - 0.5) / 255.0  # dither
        return out.clamp(0, 1)


def self_ut(disk: Disk, r):
    from .emission import kepler_ut

    return kepler_ut(torch.clamp(r, min=disk.r_isco), disk.a)


def gaussian_blur(img, sigma):
    k = int(math.ceil(3 * sigma))
    x = torch.arange(-k, k + 1, device=img.device, dtype=img.dtype)
    w = torch.exp(-0.5 * (x / sigma) ** 2)
    w = w / w.sum()
    C = img.shape[0]
    t = img[None]
    t = Fn.conv2d(Fn.pad(t, (k, k, 0, 0), mode="reflect"), w.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    t = Fn.conv2d(Fn.pad(t, (0, 0, k, k), mode="reflect"), w.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return t[0]


def filmic(img, white=6.0):
    """Hue-preserving tone map: extended Reinhard on luminance, then a luminance-preserving
    desaturation towards white only where a channel would clip (hot gas turns white-hot,
    warm gas keeps its blackbody colour)."""
    Y = (0.2126 * img[0] + 0.7152 * img[1] + 0.0722 * img[2]).clamp(min=0)
    Yd = Y * (1 + Y / (white * white)) / (1 + Y)
    Yd = Yd.clamp(max=1.0)
    rgb = img * (Yd / Y.clamp(min=1e-10))[None]
    mx = rgb.max(0).values
    t = ((mx - 1) / (mx - Yd).clamp(min=1e-6)).clamp(0, 1)
    rgb = rgb + t[None] * (Yd[None] - rgb)
    return rgb.clamp(0, 1)


def bloom(img):
    """Energy-spreading glow: a sum of progressively wider blurs (computed on a pyramid)."""
    C, H, W = img.shape
    acc = torch.zeros_like(img)
    cur = img[None]
    weights = [0.45, 0.28, 0.15, 0.08, 0.04]
    for i, w in enumerate(weights):
        cur = Fn.avg_pool2d(cur, 2)
        bl = gaussian_blur(cur[0], 1.5)[None]
        acc += w * Fn.interpolate(bl, size=(H, W), mode="bilinear", align_corners=False)[0]
    return acc


_ACES_IN = torch.tensor([[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]])
_ACES_OUT = torch.tensor([[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]])


def aces(img):
    """Stephen Hill's fit of the ACES RRT+ODT (linear sRGB in, display-linear out)."""
    Mi = _ACES_IN.to(img.device)
    Mo = _ACES_OUT.to(img.device)
    v = torch.einsum("ij,jhw->ihw", Mi, img)
    v = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081)
    return torch.einsum("ij,jhw->ihw", Mo, v).clamp(0, 1)


def srgb_oetf(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.clamp(x, min=1e-8) ** (1 / 2.4) - 0.055)


def to_uint8(img):
    return (img.permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
