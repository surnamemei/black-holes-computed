"""A simulated Event Horizon Telescope observation of M87* at 230 GHz (1.3 mm).

* 2017 EHT stations (geodetic positions; SEFDs roughly following EHT 2019 Paper III).
* Earth-rotation synthesis: (u, v) from station ECEF baselines, GMST and the source
  position (Thompson, Moran & Swenson eq. 4.1), with a 10-degree elevation limit.
* Thermal noise from the radiometer equation, a 1% systematic term, and random
  per-station, per-scan atmospheric phases plus amplitude gain errors.
* Closure phases and log closure amplitudes are immune to station-based errors; the
  regularized-maximum-likelihood imager (src/vlbi/imaging.py) uses only those.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

C = 299792458.0
FREQ = 230e9
LAM = C / FREQ
UAS = math.pi / 180 / 3600 / 1e6  # microarcsecond in radians

# name, label, lat, lon, height [m], SEFD [Jy]
STATIONS = [
    ("ALMA", "ALMA, Chile", -23.0293, -67.7549, 5074.0, 90.0),
    ("APEX", "APEX, Chile", -23.0058, -67.7591, 5104.0, 3500.0),
    ("JCMT", "JCMT, Hawaii", 19.8228, -155.4770, 4120.0, 6000.0),
    ("SMA", "SMA, Hawaii", 19.8243, -155.4782, 4115.0, 4900.0),
    ("SMT", "SMT, Arizona", 32.7016, -109.8912, 3185.0, 5000.0),
    ("LMT", "LMT, Mexico", 18.9858, -97.3147, 4593.0, 4500.0),
    ("IRAM", "IRAM 30 m, Spain", 37.0661, -3.3925, 2850.0, 1400.0),
    ("SPT", "SPT, South Pole", -89.9900, -45.0000, 2816.0, 5000.0),
]

M87_RA = math.radians((12 + 30 / 60 + 49.42338 / 3600) * 15)
M87_DEC = math.radians(12 + 23 / 60 + 28.0439 / 3600)


def geodetic_to_ecef(lat, lon, h):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    la, lo = math.radians(lat), math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(la) ** 2)
    return np.array([(N + h) * math.cos(la) * math.cos(lo), (N + h) * math.cos(la) * math.sin(lo), (N * (1 - e2) + h) * math.sin(la)])


def gmst_rad(jd):
    D = jd - 2451545.0
    return math.radians((18.697374558 + 24.06570982441908 * D) % 24 * 15)


@dataclass
class Observation:
    t_hours: np.ndarray  # (K,) scan times, hours UTC
    i: np.ndarray  # (K,) station index 1
    j: np.ndarray  # (K,) station index 2
    scan: np.ndarray  # (K,) scan index
    u: np.ndarray  # (K,) wavelengths
    v: np.ndarray
    vis_true: np.ndarray  # complex Jy
    vis_obs: np.ndarray  # complex Jy (noise + gains)
    sigma: np.ndarray  # thermal+systematic sigma, Jy
    station_names: list
    elev: np.ndarray  # (n_scans, n_stations) degrees
    scan_times: np.ndarray


def elevation(st_ecef_latlon, gmst, ra, dec):
    lat, lon = st_ecef_latlon
    h = gmst + math.radians(lon) - ra  # local hour angle
    la = math.radians(lat)
    s = math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(h)
    return math.degrees(math.asin(max(-1, min(1, s))))


def uv_for(bl, H, dec):
    """TMS eq. 4.1 with H the Greenwich hour angle of the source; bl = ECEF baseline / lambda."""
    Lx, Ly, Lz = bl
    u = math.sin(H) * Lx + math.cos(H) * Ly
    v = -math.sin(dec) * math.cos(H) * Lx + math.sin(dec) * math.sin(H) * Ly + math.cos(dec) * Lz
    return u, v


def schedule(jd0=2457854.5, hours=24.0, cadence_min=6.0, el_min=10.0, ra=M87_RA, dec=M87_DEC, stations=STATIONS):
    """Baselines visible at each scan (both stations above el_min)."""
    ecef = [geodetic_to_ecef(s[2], s[3], s[4]) for s in stations]
    times = np.arange(0, hours, cadence_min / 60)
    rows = []
    elev = np.zeros((len(times), len(stations)))
    for k, t in enumerate(times):
        gm = gmst_rad(jd0 + t / 24)
        H = gm - ra
        up = []
        for n, s in enumerate(stations):
            e = elevation((s[2], s[3]), gm, ra, dec)
            elev[k, n] = e
            if e > el_min:
                up.append(n)
        for a, b in itertools.combinations(up, 2):
            bl = (ecef[b] - ecef[a]) / LAM
            u, v = uv_for(bl, H, dec)
            rows.append((t, a, b, k, u, v))
    r = np.array(rows)
    return dict(t=r[:, 0], i=r[:, 1].astype(int), j=r[:, 2].astype(int), scan=r[:, 3].astype(int), u=r[:, 4], v=r[:, 5],
                elev=elev, times=times)


def dft_matrix(u, v, n, pix_rad):
    """A[k, p] = exp(-2 pi i (u_k x_p + v_k y_p)); x = RA offset (east = +x, drawn to the left), y = Dec."""
    c = (np.arange(n) - (n - 1) / 2) * pix_rad
    # image row 0 is north (top); column 0 is east (left), as on the sky
    X, Y = np.meshgrid(-c, -c)
    return np.exp(-2j * np.pi * (np.outer(u, X.ravel()) + np.outer(v, Y.ravel())))


def observe(img, pix_uas, sched, seed=1, t_int=300.0, bw=4e9, eta=0.88, sys_frac=0.01, gain_amp=0.1,
            stations=STATIONS):
    """Simulate visibilities of image `img` (n x n, Jy/pixel) for a schedule."""
    rng = np.random.default_rng(seed)
    n = img.shape[0]
    A = dft_matrix(sched["u"], sched["v"], n, pix_uas * UAS)
    V = A @ img.ravel()
    sefd = np.array([s[5] for s in stations])
    sig_th = (1 / eta) * np.sqrt(sefd[sched["i"]] * sefd[sched["j"]] / (2 * bw * t_int))
    sig = np.sqrt(sig_th ** 2 + (sys_frac * np.abs(V)) ** 2)
    noise = (rng.normal(size=V.shape) + 1j * rng.normal(size=V.shape)) * sig / math.sqrt(2)
    n_scan = sched["scan"].max() + 1
    ph = rng.uniform(-np.pi, np.pi, size=(n_scan, len(stations)))  # atmosphere, random every scan
    amp = np.exp(rng.normal(0, gain_amp, size=len(stations)))  # amplitude gain errors
    g_i = amp[sched["i"]] * np.exp(1j * ph[sched["scan"], sched["i"]])
    g_j = amp[sched["j"]] * np.exp(1j * ph[sched["scan"], sched["j"]])
    V_obs = g_i * np.conj(g_j) * (V + noise)
    return V, V_obs, sig


def closure_phases(sched, V, sig):
    """All triangles per scan: phi_ij + phi_jk - phi_ik; returns index triples into V and sigma."""
    tri, sgn = [], []
    key = {(sched["scan"][k], sched["i"][k], sched["j"][k]): k for k in range(len(V))}
    for s in np.unique(sched["scan"]):
        sts = sorted(set(sched["i"][sched["scan"] == s]) | set(sched["j"][sched["scan"] == s]))
        for a, b, c in itertools.combinations(sts, 3):
            k1, k2, k3 = key.get((s, a, b)), key.get((s, b, c)), key.get((s, a, c))
            if None not in (k1, k2, k3):
                tri.append((k1, k2, k3))
    tri = np.array(tri)
    cp = np.angle(V[tri[:, 0]] * V[tri[:, 1]] * np.conj(V[tri[:, 2]]))
    s_cp = np.sqrt((sig[tri[:, 0]] / np.abs(V[tri[:, 0]])) ** 2 + (sig[tri[:, 1]] / np.abs(V[tri[:, 1]])) ** 2
                   + (sig[tri[:, 2]] / np.abs(V[tri[:, 2]])) ** 2)
    return tri, cp, s_cp


def closure_amps(sched, V, sig):
    """Quadrangles per scan: log |V_ab V_cd / (V_ac V_bd)|."""
    quads = []
    key = {(sched["scan"][k], sched["i"][k], sched["j"][k]): k for k in range(len(V))}
    for s in np.unique(sched["scan"]):
        sts = sorted(set(sched["i"][sched["scan"] == s]) | set(sched["j"][sched["scan"] == s]))
        for a, b, c, d in itertools.combinations(sts, 4):
            for (p, q, r, t) in ((a, b, c, d), (a, c, b, d)):
                ks = [key.get((s, *sorted(x))) for x in ((p, q), (r, t), (p, r), (q, t))]
                if None not in ks:
                    quads.append(ks)
    quads = np.array(quads)
    lca = (np.log(np.abs(V[quads[:, 0]])) + np.log(np.abs(V[quads[:, 1]]))
           - np.log(np.abs(V[quads[:, 2]])) - np.log(np.abs(V[quads[:, 3]])))
    s_lca = np.sqrt(sum((sig[quads[:, m]] / np.abs(V[quads[:, m]])) ** 2 for m in range(4)))
    return quads, lca, s_lca


def dirty_image(sched, V, n, pix_uas, weights=None):
    A = dft_matrix(sched["u"], sched["v"], n, pix_uas * UAS)
    w = np.ones(len(V)) if weights is None else weights
    img = np.real(np.conj(A).T @ (w * V)).reshape(n, n) / w.sum()
    beam = np.real(np.conj(A).T @ w).reshape(n, n) / w.sum()
    return img, beam
