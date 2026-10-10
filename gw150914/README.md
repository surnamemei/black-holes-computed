# GW150914, found again

The first gravitational-wave detection, recovered from LIGO's public O1 strain with a from-scratch matched-filter search. NumPy and SciPy do all the work; LALSimulation supplies the IMRPhenomD waveforms. The interactive page is https://surnamemei.github.io/black-holes-computed/hear/ (source in `web/src/`, built into `../docs/hear/` by `../tools/make_pages.py`). Outputs used by the page are in `results/`.

## Results vs published

| Quantity | This analysis | LIGO/Virgo |
|---|---|---|
| Network SNR | 23.89 | 24 |
| Hanford / Livingston SNR | 19.80 / 13.36 | ≈ 20 / ≈ 13 |
| Arrival difference | 7.08 ms, Livingston first | 6.9 (+0.5/−0.4) ms |
| Phase difference | 167° | ≈ 180° (antenna orientation) |
| χ² per d.o.f. at the event | 1.10 Hanford, 0.43 Livingston | (signal-like ≈ 1) |
| Detector-frame chirp mass | 31.6 [30.9, 32.2] M☉ | ≈ 31 M☉ (GWTC-1 source frame 28.6, z ≈ 0.09) |
| Background, 1,000 time slides (43.7 days) | loudest raw 16.25 (one Livingston glitch); after χ² reweighting 7.38 | FAR < 1 / 203,000 yr |

## Pipeline

1. **`analysis.py`.** This step does four things:
   - median-averaged Welch PSDs from 4,096 s per detector;
   - a frequency-domain complex matched filter from 30 to 1,024 Hz, run over 1,066 non-spinning IMRPhenomD templates (20–60 and 15–60 M☉, detector frame);
   - a two-detector coincidence within 10 ms;
   - a chirp-mass estimate from exp(ρ²/2) over the bank.
2. **`products.py`.** Produces the Q-transform, whitened traces, best-template overlays, audio, and the page data. The audio is offered at true pitch and with a +400 Hz single-sideband shift.
3. **`chisq.py`.** Applies Allen's χ² test over 16 frequency bins of equal expected SNR² and the PyCBC reweighted SNR, then repeats the time slides with the reweighted statistic.

Run them in order with `.venv/bin/python`. The data are the two `*_LOSC_4_V1-1126256640-4096.hdf5` files from GWOSC, stored in `data/`.

## Lessons recorded along the way

- **The glitch is in "good" data.** Livingston contains a glitch 217 s before the event with single-detector SNR 15.45. It passes every GWOSC data-quality category, so only the χ² test removes it (χ²/dof = 15.8, reweighted SNR 4.36).
- **`np.roll` wraps around.** In time slides, the event-exclusion mask must be rolled exactly like the data. A version that excluded by unwrapped time let the real Livingston signal leak into 73 background slides (a false background of about 13.7).

Simplifications: no spins (masses biased toward equal), no full Bayesian parameter estimation, and a 44-day background.

Acknowledgment: This research has made use of data obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration and KAGRA.
