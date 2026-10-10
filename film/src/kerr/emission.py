"""Emission physics for equatorial matter around a Kerr black hole (G = c = M = 1).

* Prograde circular (Keplerian) orbits: Omega, u^t, E, L.
* Novikov-Thorne / Page-Thorne radiative flux F(r) of a thin disk (numerical integral).
* Cunningham (1975) plunging flow inside the ISCO, for optically thin models.
* Redshift factor g = nu_obs / nu_emit for each recorded equatorial crossing.
* Blackbody colours via the CIE 1931 colour-matching functions (Wyman et al. 2013 fit):
  a blackbody seen with redshift g is exactly a blackbody at T_obs = g T.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .tracer import isco_radius

# ----------------------------------------------------------------------------- orbits


def kepler_omega(r, a):
    return 1.0 / (r ** 1.5 + a)


def kepler_ut(r, a):
    return (r ** 1.5 + a) / (r ** 0.75 * torch.sqrt(torch.clamp(r ** 1.5 - 3.0 * r ** 0.5 + 2.0 * a, min=1e-12)))


def kepler_EL(r, a):
    den = r ** 0.75 * np.sqrt(r ** 1.5 - 3.0 * r ** 0.5 + 2.0 * a)
    E = (r ** 1.5 - 2.0 * r ** 0.5 + a) / den
    L = (r * r - 2.0 * a * r ** 0.5 + a * a) / den
    return E, L


def novikov_thorne_table(a, r_out=60.0, n=6000):
    """Page-Thorne flux F(r) on a grid from r_isco to r_out (float64), normalised to max = 1.

    F(r) = (Mdot / 4 pi r) * (-dOmega/dr) / (E - Omega L)^2 * Int_{r_isco}^r (E - Omega L) dL/dr dr
    """
    r_in = isco_radius(a)
    # dense grid, clustered near the ISCO where the integrand varies fastest
    x = np.linspace(0, 1, n)
    r = r_in + (r_out - r_in) * x ** 2
    E, L = kepler_EL(r, a)
    Om = 1.0 / (r ** 1.5 + a)
    dOm = np.gradient(Om, r)
    dL = np.gradient(L, r)
    integrand = (E - Om * L) * dL
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))])
    F = (1.0 / (4 * np.pi * r)) * (-dOm) / (E - Om * L) ** 2 * integral
    F = np.maximum(F, 0.0)
    return r, F, r[np.argmax(F)]


class Disk:
    """Lookup tables for an equatorial thin disk around a Kerr hole of spin a."""

    def __init__(self, a, r_out=60.0, device="cuda", flux_norm="self"):
        """flux_norm: "self" -> F / max F of this spin; "schwarzschild" -> F / max F of an a = 0
        disk with the same accretion rate (so spin-ups compare physically)."""
        self.a = a
        self.r_isco = isco_radius(a)
        r, F, r_peak = novikov_thorne_table(a, r_out)
        F = F / (F.max() if flux_norm == "self" else novikov_thorne_table(0.0, r_out)[1].max())
        self.r_peak = r_peak
        self.r_tab = torch.tensor(r, dtype=torch.float32, device=device)
        self.F_tab = torch.tensor(F, dtype=torch.float32, device=device)
        E, L = kepler_EL(np.array([self.r_isco]), a)
        self.E_isco, self.L_isco = float(E[0]), float(L[0])

    def flux(self, r):
        """Normalised Novikov-Thorne flux (0 inside the ISCO)."""
        idx = torch.searchsorted(self.r_tab, r.contiguous()).clamp(1, len(self.r_tab) - 1)
        r0, r1 = self.r_tab[idx - 1], self.r_tab[idx]
        f0, f1 = self.F_tab[idx - 1], self.F_tab[idx]
        t = ((r - r0) / (r1 - r0)).clamp(0, 1)
        F = f0 + t * (f1 - f0)
        # beyond the table, continue with the Newtonian r^-3 tail
        r_end = self.r_tab[-1]
        F = torch.where(r > r_end, self.F_tab[-1] * (r_end / r) ** 3, F)
        return torch.where(r < self.r_isco, torch.zeros_like(F), F)

    def redshift(self, r, b, p_r, E_cam):
        """g = nu_cam / nu_emit for emitters on Keplerian orbits (r >= r_isco) or
        on Cunningham plunging geodesics (r < r_isco).  p_r is the photon's covariant
        radial momentum at the crossing (E = 1 normalisation)."""
        a = self.a
        # circular orbits
        ut = kepler_ut(torch.clamp(r, min=self.r_isco), a)
        Om = kepler_omega(torch.clamp(r, min=self.r_isco), a)
        g_circ = E_cam / (ut * (1.0 - Om * b))
        # plunging region: u_t = -E_isco, u_phi = L_isco
        D = r * r - 2.0 * r + a * a
        A = (r * r + a * a) ** 2 - a * a * D
        gtt = -A / (r * r * D)
        gtp = -2.0 * a / (r * D)
        gpp = (D - a * a) / (r * r * D)
        grr = D / (r * r)
        ut_c, up_c = -self.E_isco, self.L_isco
        Ut = gtt * ut_c + gtp * up_c
        Up = gtp * ut_c + gpp * up_c
        Ur2 = grr * (-1.0 - (gtt * ut_c * ut_c + 2 * gtp * ut_c * up_c + gpp * up_c * up_c))
        Ur = -torch.sqrt(torch.clamp(Ur2, min=0.0))
        pu = -1.0 * Ut + p_r * Ur + b * Up  # p_mu u^mu (negative)
        g_plunge = E_cam / torch.clamp(-pu, min=1e-6)
        return torch.where(r >= self.r_isco, g_circ, g_plunge)

    def orbital_phase(self, r, t):
        """Azimuthal advance Omega(r) t of Keplerian gas (used to shear disk textures)."""
        return kepler_omega(torch.clamp(r, min=self.r_isco), self.a) * t


# ----------------------------------------------------------------------------- colour


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_cmf(lam_nm):
    """Wyman, Sloan & Shirley (2013) multi-lobe fit to the CIE 1931 2-degree observer."""
    l = lam_nm
    x = 1.056 * _g(l, 599.8, 37.9, 31.0) + 0.362 * _g(l, 442.0, 16.0, 26.7) - 0.065 * _g(l, 501.1, 20.4, 26.2)
    y = 0.821 * _g(l, 568.8, 46.9, 40.5) + 0.286 * _g(l, 530.9, 16.3, 31.1)
    z = 1.217 * _g(l, 437.0, 11.8, 36.0) + 0.681 * _g(l, 459.0, 26.0, 13.8)
    return x, y, z


XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


class BlackbodyLUT:
    """Linear-sRGB colour and relative luminance of a blackbody, tabulated in log T."""

    def __init__(self, t_min=300.0, t_max=2.0e5, n=4096, device="cuda"):
        lam = np.linspace(360, 830, 941) * 1e-9
        xb, yb, zb = cie_cmf(lam * 1e9)
        T = np.exp(np.linspace(math.log(t_min), math.log(t_max), n))
        h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
        with np.errstate(over="ignore"):
            B = (2 * h * c * c / lam[None, :] ** 5) / np.expm1(h * c / (lam[None, :] * k * T[:, None]))
        X = np.trapezoid(B * xb, lam, axis=1)
        Y = np.trapezoid(B * yb, lam, axis=1)
        Z = np.trapezoid(B * zb, lam, axis=1)
        rgb = np.stack([X, Y, Z], -1) @ XYZ_TO_SRGB.T
        rgb = np.maximum(rgb, 0)
        rgb_unit = rgb / np.maximum(Y[:, None], 1e-300)  # colour at unit luminance
        self.logT0, self.logT1 = math.log(t_min), math.log(t_max)
        self.n = n
        self.rgb = torch.tensor(rgb_unit, dtype=torch.float32, device=device)
        self.logY = torch.tensor(np.log(np.maximum(Y, 1e-300)), dtype=torch.float32, device=device)

    def lookup(self, T):
        """Returns (rgb at unit luminance, luminance) for temperatures T (tensor)."""
        T = torch.nan_to_num(T, nan=1.0, posinf=1e9, neginf=1.0)
        u = (torch.log(torch.clamp(T, min=1.0)) - self.logT0) / (self.logT1 - self.logT0) * (self.n - 1)
        u = u.clamp(0, self.n - 1.001)
        i = u.floor().long()
        f = (u - i)[..., None]
        rgb = self.rgb[i] * (1 - f) + self.rgb[i + 1] * f
        logY = self.logY[i] * (1 - f[..., 0]) + self.logY[i + 1] * f[..., 0]
        return rgb, torch.exp(logY)
