"""Procedural celestial sphere (stars + a dusty galactic band) as a mip-mapped
equirectangular HDR texture, sampled with ray-footprint-aware filtering so the
violently magnified/demagnified sky near the photon ring does not alias.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .emission import BlackbodyLUT


def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).float() / float(0xFFFFFF)


def value_noise3(p, seed=0):
    """Smooth value noise on a 3D lattice; p: (..., 3) float tensor. Returns [0, 1]."""
    pf = torch.floor(p)
    f = p - pf
    i = pf.long()
    u = f * f * (3 - 2 * f)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1]) * (u[..., 2] if dz else 1 - u[..., 2])
                out = out + w * _hash3(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


def fbm3(p, octaves=6, seed=0, lacunarity=2.03, gain=0.5):
    amp, tot, norm = 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * value_noise3(p, seed + 17 * o)
        norm += amp
        amp *= gain
        p = p * lacunarity
    return tot / norm


class Sky:
    def __init__(self, width=16384, seed=7, n_stars=2_000_000, band_tilt_deg=62.0, band_node_deg=25.0,
                 band_strength=1.0, star_strength=1.0, device="cuda"):
        self.W, self.H = width, width // 2
        g = torch.Generator(device=device).manual_seed(seed)
        W, H = self.W, self.H
        # texel-centre directions
        lon = (torch.arange(W, device=device) + 0.5) / W * 2 * math.pi - math.pi
        lat = math.pi / 2 - (torch.arange(H, device=device) + 0.5) / H * math.pi
        # galactic frame: rotate so the band is a tilted great circle
        ti, no = math.radians(band_tilt_deg), math.radians(band_node_deg)
        Rz = torch.tensor([[math.cos(no), -math.sin(no), 0], [math.sin(no), math.cos(no), 0], [0, 0, 1]], device=device)
        Rx = torch.tensor([[1, 0, 0], [0, math.cos(ti), -math.sin(ti)], [0, math.sin(ti), math.cos(ti)]], device=device)
        self.R_gal = (Rx @ Rz).float()  # world -> galactic
        img = torch.zeros(3, H, W, device=device, dtype=torch.float32)
        rows = 512
        for r0 in range(0, H, rows):
            la = lat[r0:r0 + rows]
            LA, LO = torch.meshgrid(la, lon, indexing="ij")
            d = torch.stack([torch.cos(LA) * torch.cos(LO), torch.cos(LA) * torch.sin(LO), torch.sin(LA)], -1)
            dg = d @ self.R_gal.T
            b = torch.asin(dg[..., 2].clamp(-1, 1))  # galactic latitude
            l = torch.atan2(dg[..., 1], dg[..., 0])
            # diffuse band: thin disk + bulge towards l = 0, modulated by turbulence
            core = torch.exp(-0.5 * (b / 0.065) ** 2) * (0.5 + 0.5 * torch.exp(-0.5 * (l / 0.9) ** 2))
            halo = 0.25 * torch.exp(-0.5 * (b / 0.16) ** 2)
            bulge = 0.8 * torch.exp(-0.5 * ((b / 0.14) ** 2 + (l / 0.25) ** 2))
            n1 = fbm3(dg * 9.0, octaves=8, seed=seed)
            n2 = fbm3(dg * 40.0 + 3.1, octaves=5, seed=seed + 5)
            glow = (core + halo + bulge) * (0.15 + 1.5 * n1 ** 3.0) * (0.7 + 0.6 * n2)
            # dust lanes: dark filaments hugging the mid-plane
            lane = torch.exp(-0.5 * (b / 0.035) ** 2) * torch.clamp(1.8 * fbm3(dg * 24.0 + 7.7, 6, seed + 9) - 0.5, 0, 1)
            dust = torch.clamp(1 - 0.95 * lane, 0.03, 1.0)
            glow = glow * dust
            # colour: warm bulge, bluish arms
            warm = torch.tensor([1.0, 0.78, 0.55], device=device)[:, None, None]
            cool = torch.tensor([0.80, 0.85, 1.0], device=device)[:, None, None]
            mix = torch.exp(-0.5 * (l / 0.8) ** 2)[None]
            col = warm * mix + cool * (1 - mix)
            img[:, r0:r0 + rows] = band_strength * 0.016 * glow[None] * col
            # faint all-sky background
            img[:, r0:r0 + rows] += 0.00015
        # stars: concentrate towards the band (half isotropic, half disk-like)
        bb = BlackbodyLUT(device=device)
        n_iso = n_stars // 2
        z = torch.rand(n_iso, generator=g, device=device) * 2 - 1
        ph = torch.rand(n_iso, generator=g, device=device) * 2 * math.pi
        d_iso = torch.stack([torch.sqrt(1 - z * z) * torch.cos(ph), torch.sqrt(1 - z * z) * torch.sin(ph), z], -1)
        n_disk = n_stars - n_iso
        bl = torch.randn(n_disk, generator=g, device=device) * 0.14
        ll = torch.rand(n_disk, generator=g, device=device) * 2 * math.pi
        d_gal = torch.stack([torch.cos(bl) * torch.cos(ll), torch.cos(bl) * torch.sin(ll), torch.sin(bl)], -1)
        d_disk = d_gal @ self.R_gal  # galactic -> world
        dirs = torch.cat([d_iso, d_disk])
        # flux: Euclidean number counts N(>F) ~ F^-1.5
        u = torch.rand(n_stars, generator=g, device=device)
        flux = 0.015 * (u.clamp(min=1e-7)) ** (-1 / 1.5)
        flux = flux.clamp(max=300.0)
        T = 2800 + 9000 * torch.rand(n_stars, generator=g, device=device) ** 2.5
        T = torch.where(torch.rand(n_stars, generator=g, device=device) < 0.04, 15000 + 15000 * torch.rand(n_stars, generator=g, device=device), T)
        rgb, _ = bb.lookup(T)
        rgb = rgb / rgb.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        # splat each star as a small Gaussian (sigma 0.55 texel, widened in x near the poles)
        slat = torch.asin(dirs[:, 2].clamp(-1, 1))
        slon = torch.atan2(dirs[:, 1], dirs[:, 0])
        px = (slon + math.pi) / (2 * math.pi) * W - 0.5
        py = (math.pi / 2 - slat) / math.pi * H - 0.5
        sx = 0.55 / torch.cos(slat).clamp(min=0.02)
        sy = torch.full_like(sx, 0.55)
        R = 3
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                ix = (px.round() + dx).long()
                iy = (py.round() + dy).long()
                w = torch.exp(-0.5 * (((ix.float() - px) / sx) ** 2 + ((iy.float() - py) / sy) ** 2))
                w = w / (2 * math.pi * sx * sy)
                ok = (iy >= 0) & (iy < H)
                ixw = ix % W
                lin = (iy.clamp(0, H - 1) * W + ixw)[ok]
                for c in range(3):
                    img[c].view(-1).index_add_(0, lin, (star_strength * flux * rgb[:, c] * w)[ok])
        self.base = img
        self._build_mips()

    def add_galaxy(self, direction, size_deg=4.0, tilt_deg=35.0, pa_deg=20.0, brightness=0.6, arms=2, seed=3):
        """Paint a small procedural spiral galaxy centred on a world direction (a lensing target)."""
        d0 = torch.tensor(direction, dtype=torch.float32, device=self.base.device)
        d0 = d0 / d0.norm()
        W, H = self.W, self.H
        lat0 = math.asin(float(d0[2]))
        span = math.radians(size_deg) * 1.6
        r0 = max(int((math.pi / 2 - lat0 - span) / math.pi * H), 0)
        r1 = min(int((math.pi / 2 - lat0 + span) / math.pi * H) + 1, H)
        lat = math.pi / 2 - (torch.arange(r0, r1, device=d0.device) + 0.5) / H * math.pi
        lon = (torch.arange(W, device=d0.device) + 0.5) / W * 2 * math.pi - math.pi
        LA, LO = torch.meshgrid(lat, lon, indexing="ij")
        d = torch.stack([torch.cos(LA) * torch.cos(LO), torch.cos(LA) * torch.sin(LO), torch.sin(LA)], -1)
        # tangent-plane coordinates around d0
        up = torch.tensor([0.0, 0.0, 1.0], device=d0.device)
        e1 = torch.linalg.cross(up, d0); e1 = e1 / e1.norm()
        e2 = torch.linalg.cross(d0, e1)
        cosang = (d * d0).sum(-1)
        x = (d * e1).sum(-1) / cosang.clamp(min=1e-3)
        y = (d * e2).sum(-1) / cosang.clamp(min=1e-3)
        pa = math.radians(pa_deg)
        xr = x * math.cos(pa) + y * math.sin(pa)
        yr = (-x * math.sin(pa) + y * math.cos(pa)) / math.cos(math.radians(tilt_deg))
        R = torch.sqrt(xr ** 2 + yr ** 2) / math.radians(size_deg / 2)
        th = torch.atan2(yr, xr)
        disk = torch.exp(-R / 0.32)
        spiral = 0.5 + 0.5 * torch.cos(arms * (th - 2.2 * torch.log(R.clamp(min=0.05))))
        knots = fbm3(torch.stack([xr, yr, torch.zeros_like(xr)], -1) * (60.0 / math.radians(size_deg)), 4, seed)
        bulge = torch.exp(-(R / 0.09) ** 2)
        lum = disk * (0.25 + 0.75 * spiral ** 2) * (0.6 + 0.8 * knots) + 2.5 * bulge
        lum = lum * (cosang > 0).float() * (R < 1.4).float()
        col_d = torch.tensor([0.75, 0.82, 1.0], device=d0.device)[:, None, None]
        col_b = torch.tensor([1.0, 0.85, 0.6], device=d0.device)[:, None, None]
        mixb = (bulge / (bulge + disk * 0.5 + 1e-6))[None]
        self.base[:, r0:r1] += brightness * lum[None] * (col_b * mixb + col_d * (1 - mixb))
        self._build_mips()

    def _build_mips(self):
        img = self.base
        # mip pyramid, stored pre-padded (longitude wrap) in float32: grid_sample
        # coordinates in fp16 would misplace stars by ~8 texels at this width.
        self.pad = 2
        levels = [img]
        cur = img
        while cur.shape[-1] > 64:
            cur = Fn.avg_pool2d(cur[None], 2)[0]
            levels.append(cur)
        self.levels = [torch.cat([t[..., -self.pad:], t, t[..., :self.pad]], -1)[None].contiguous() for t in levels]

    def sample(self, d, footprint):
        """d: (N, 3) unit directions; footprint: (N,) angular size (rad) of each sample.
        Trilinear mip filtering in an equirectangular map with longitude wrap."""
        lat = torch.asin(d[:, 2].clamp(-1, 1))
        lon = torch.atan2(d[:, 1], d[:, 0])
        texel0 = 2 * math.pi / self.W
        lvl = torch.log2((footprint / texel0).clamp(min=1.0))
        lvl = lvl.clamp(0, len(self.levels) - 1.001)
        out = torch.zeros(d.shape[0], 3, device=d.device)
        x = lon / math.pi  # [-1, 1]
        y = -lat / (math.pi / 2)
        grid = torch.stack([x, y], -1)[None, None].float()
        lo = lvl.floor().long()
        frac = (lvl - lo)
        for L in range(len(self.levels)):
            m0 = lo == L
            m1 = (lo + 1) == L
            m = m0 | m1
            if not m.any():
                continue
            tex = self.levels[L]
            pad = self.pad
            Wl = tex.shape[-1] - 2 * pad
            g = grid[:, :, m].clone()
            g[..., 0] = g[..., 0] * Wl / (Wl + 2 * pad)
            s = Fn.grid_sample(tex, g, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
            w = torch.where(m0[m], 1 - frac[m], frac[m])[:, None]
            out[m] += w * s
        return out
