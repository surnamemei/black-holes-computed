"""Regularized maximum-likelihood (RML) imaging from closure quantities only.

Objective (cf. Chael et al. 2018; EHT 2019 Paper IV):
    J(I) = a_cp chi2_cphase + a_lca chi2_logcamp + b_mem S_rel(I | prior)
           + b_tsv TSV(I) + b_flux (sum I - F)^2 / F^2
with I = exp(x) > 0, optimized by L-BFGS in rounds; each round restarts from the
previous image blurred by a fraction of the nominal beam.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .eht import UAS, dft_matrix


def gaussian_image(n, pix_uas, fwhm_uas, flux):
    c = (np.arange(n) - (n - 1) / 2) * pix_uas
    X, Y = np.meshgrid(c, c)
    s = fwhm_uas / 2.3548
    g = np.exp(-0.5 * (X ** 2 + Y ** 2) / s ** 2)
    return g / g.sum() * flux


def blur(img, pix_uas, fwhm_uas):
    if fwhm_uas <= 0:
        return img
    n = img.shape[0]
    f = np.fft.fftfreq(n, d=pix_uas)
    FX, FY = np.meshgrid(f, f)
    s = fwhm_uas / 2.3548
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * np.pi ** 2 * s ** 2 * (FX ** 2 + FY ** 2))))


class RMLImager:
    def __init__(self, sched, tri, cp, s_cp, quads, lca, s_lca, n=64, pix_uas=2.5, flux=0.6, prior_fwhm=55.0,
                 device="cuda"):
        self.n, self.pix, self.flux = n, pix_uas, flux
        A = dft_matrix(sched["u"], sched["v"], n, pix_uas * UAS)
        self.A = torch.tensor(A, dtype=torch.complex128, device=device)
        t = lambda x, dt=torch.float64: torch.tensor(x, dtype=dt, device=device)
        self.tri, self.cp, self.s_cp = t(tri, torch.long), t(cp), t(s_cp)
        self.quads, self.lca, self.s_lca = t(quads, torch.long), t(lca), t(s_lca)
        self.prior = t(gaussian_image(n, pix_uas, prior_fwhm, flux)).clamp(min=1e-12)
        self.device = device

    def terms(self, I):
        V = self.A @ I.reshape(-1).to(torch.complex128)
        cpm = torch.angle(V[self.tri[:, 0]] * V[self.tri[:, 1]] * torch.conj(V[self.tri[:, 2]]))
        chi_cp = 2.0 * torch.mean((1 - torch.cos(self.cp - cpm)) / self.s_cp ** 2)
        lV = torch.log(torch.abs(V).clamp(min=1e-12))
        lcam = lV[self.quads[:, 0]] + lV[self.quads[:, 1]] - lV[self.quads[:, 2]] - lV[self.quads[:, 3]]
        chi_lca = torch.mean(((self.lca - lcam) / self.s_lca) ** 2)
        F = I.sum()
        mem = torch.sum(I * torch.log(I / self.prior)) / F
        dx = I[:, 1:] - I[:, :-1]
        dy = I[1:, :] - I[:-1, :]
        tsv = (torch.sum(dx ** 2) + torch.sum(dy ** 2)) / (self.flux ** 2) * self.n ** 2 / 4.0
        fl = ((F - self.flux) / self.flux) ** 2
        return chi_cp, chi_lca, mem, tsv, fl

    def run(self, init=None, rounds=((10, 10, 1.0, 1.0, 1e3, 400, 0.0),), snapshot_every=2, callback=None):
        """rounds: tuples (a_cp, a_lca, b_mem, b_tsv, b_flux, iters, blur_frac_of_beam_before_round)."""
        I0 = self.prior.clone() if init is None else torch.tensor(init, dtype=torch.float64, device=self.device)
        snaps = [I0.cpu().numpy()]
        history = []
        beam_uas = 20.0
        cur = I0
        for (a_cp, a_lca, b_mem, b_tsv, b_flux, iters, blur_frac) in rounds:
            if blur_frac > 0:
                cur = torch.tensor(blur(cur.cpu().numpy(), self.pix, blur_frac * beam_uas), dtype=torch.float64,
                                   device=self.device).clamp(min=1e-10)
            x = torch.log(cur.clamp(min=1e-10)).detach().requires_grad_(True)
            opt = torch.optim.LBFGS([x], lr=1, max_iter=1, history_size=20, line_search_fn="strong_wolfe")
            it = [0]

            def closure():
                opt.zero_grad()
                I = torch.exp(x)
                c1, c2, m, tv, fl = self.terms(I)
                J = a_cp * c1 + a_lca * c2 + b_mem * m + b_tsv * tv + b_flux * fl
                J.backward()
                return J

            for k in range(iters):
                opt.step(closure)
                with torch.no_grad():
                    I = torch.exp(x)
                    if (k % snapshot_every) == 0:
                        snaps.append(I.cpu().numpy())
                        c1, c2, m, tv, fl = self.terms(I)
                        history.append(dict(chi_cp=c1.item(), chi_lca=c2.item(), mem=m.item(), tsv=tv.item(), flux=I.sum().item()))
                        if callback:
                            callback(len(snaps), history[-1])
            cur = torch.exp(x).detach()
        return cur.cpu().numpy(), snaps, history


def nxcorr(a, b):
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())
