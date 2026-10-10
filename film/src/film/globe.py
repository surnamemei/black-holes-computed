"""Orthographic Earth globe from Natural Earth land polygons (public domain), rendered per pixel on the GPU."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageDraw

from .core import DEV, H, ROOT, W

_MASK = None


def land_mask(width=4096):
    global _MASK
    if _MASK is not None:
        return _MASK
    cache = os.path.join(ROOT, "out", "cache", f"landmask_{width}.npy")
    if os.path.exists(cache):
        m = np.load(cache)
    else:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        hgt = width // 2
        img = Image.new("L", (width, hgt), 0)
        d = ImageDraw.Draw(img)
        geo = json.load(open(os.path.join(ROOT, "assets", "naturalearth", "ne_50m_land.geojson")))
        for f in geo["features"]:
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                for k, ring in enumerate(poly):
                    pts = [((lon + 180) / 360 * width, (90 - lat) / 180 * hgt) for lon, lat in ring]
                    d.polygon(pts, fill=255 if k == 0 else 0)
        m = np.asarray(img, dtype=np.float32) / 255.0
        np.save(cache, m)
    _MASK = torch.tensor(m, device=DEV)[None, None]
    return _MASK


def rot_matrix(lat0, lon0):
    """Rows: view axes (right, up, toward viewer) expressed in Earth-fixed ECEF unit vectors."""
    la, lo = math.radians(lat0), math.radians(lon0)
    toward = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.cross(toward, east)
    return np.stack([east, north, toward])


class Globe:
    def __init__(self):
        self.mask = land_mask()

    def project(self, lat, lon, lat0, lon0, R, cx, cy, alt=0.0):
        """Screen position (x, y) and depth z (>0 = front hemisphere) of a lat/lon point."""
        M = rot_matrix(lat0, lon0)
        la, lo = math.radians(lat), math.radians(lon)
        p = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]) * (1 + alt)
        v = M @ p
        return cx + R * v[0], cy - R * v[1], v[2]

    def project_xyz(self, P, lat0, lon0, R, cx, cy):
        M = rot_matrix(lat0, lon0)
        v = P @ M.T
        return np.stack([cx + R * v[:, 0], cy - R * v[:, 1], v[:, 2]], -1)

    def render(self, lat0, lon0, R=400.0, cx=W / 2, cy=H / 2, sun=(-0.5, 0.55, 0.65), bg=None, alpha=1.0,
               land_col=(0.13, 0.20, 0.30), ocean_col=(0.02, 0.05, 0.10), coast_col=(0.35, 0.75, 1.0), grid=0.10):
        """Returns (rgb (3,H,W), coverage (1,H,W))."""
        yy, xx = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32) + 0.5,
                                torch.arange(W, device=DEV, dtype=torch.float32) + 0.5, indexing="ij")
        x = (xx - cx) / R
        y = (cy - yy) / R
        r2 = x * x + y * y
        inside = r2 < 1.0
        z = torch.sqrt(torch.clamp(1 - r2, min=0))
        M = torch.tensor(rot_matrix(lat0, lon0), dtype=torch.float32, device=DEV)
        v = torch.stack([x, y, z], -1)  # view coords
        p = v @ M  # ECEF (since M rows are the view axes in ECEF)
        lat = torch.asin(p[..., 2].clamp(-1, 1))
        lon = torch.atan2(p[..., 1], p[..., 0])
        grid_xy = torch.stack([lon / math.pi, -lat / (math.pi / 2)], -1)[None]
        land = Fn.grid_sample(self.mask, grid_xy, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
        # coastline: gradient magnitude of the smoothed land mask
        lp = Fn.avg_pool2d(land[None, None], 3, 1, 1)[0, 0]
        gx = torch.zeros_like(lp); gy = torch.zeros_like(lp)
        gx[:, 1:-1] = lp[:, 2:] - lp[:, :-2]
        gy[1:-1, :] = lp[2:, :] - lp[:-2, :]
        coast = torch.sqrt(gx * gx + gy * gy).clamp(0, 1)
        # lighting (soft terminator) in view space
        s = torch.tensor(sun, device=DEV); s = s / s.norm()
        lam = (v * s).sum(-1)
        light = 0.35 + 0.65 * torch.sigmoid(lam * 6)
        lc = torch.tensor(land_col, device=DEV).view(3, 1, 1)
        oc = torch.tensor(ocean_col, device=DEV).view(3, 1, 1)
        cc = torch.tensor(coast_col, device=DEV).view(3, 1, 1)
        col = (oc * (1 - land) + lc * land) * light + cc * coast * 1.3 * (0.6 + 0.4 * light)
        # graticule every 30 degrees
        if grid > 0:
            glat = torch.abs(torch.remainder(torch.rad2deg(lat) + 15, 30) - 15)
            glon = torch.abs(torch.remainder(torch.rad2deg(lon) + 15, 30) - 15)
            pxdeg = R * math.radians(1.0)
            gl = (torch.clamp(1 - glat * pxdeg / 1.2, 0, 1) + torch.clamp(1 - glon * pxdeg * torch.cos(lat) / 1.2, 0, 1)).clamp(0, 1)
            col = col + grid * gl * cc * 0.5
        # limb darkening + atmosphere rim
        col = col * (0.55 + 0.45 * z)
        rr = torch.sqrt(r2)
        rim = torch.exp(-((rr - 1.0) / 0.035) ** 2) * 0.55 + torch.exp(-((rr - 1.0) / 0.12) ** 2) * 0.18
        rimcol = torch.tensor((0.30, 0.60, 1.0), device=DEV).view(3, 1, 1)
        edge = torch.clamp((1 - rr) * R / 1.5, 0, 1)  # antialiased disk edge
        out = col * edge + rimcol * rim * (1 - 0.6 * edge)
        cov = torch.clamp(edge + rim, 0, 1)
        return out * alpha, cov[None] * alpha
