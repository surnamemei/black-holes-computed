# How to Photograph a Black Hole

A 5½-minute narrated science film in which **every frame is computed**. It covers:

1. What a black hole looks like, from a Kerr-metric ray tracer written for this project.
2. How the Event Horizon Telescope photographed one, from a simulated 2017 EHT observation and a regularized image reconstruction that uses closure quantities.

It was made in one session on a single RTX 5090. Claude chose and built it after being asked to "decide what you actually want to do… a complex and valuable project". Run every command below from this `film/` directory.

| Deliverable | Where |
|---|---|
| Film (1080p30, H.264 + AAC, soft subtitles) | the GitHub release of this repository |
| Web version (720p chapters) and the interactive pages | `../docs/photograph/` |
| Subtitles | `results/film.srt` |
| Physics validation | `results/validation/` (figures + `validation_results.json`) |
| EHT simulation diagnostics | `results/vlbi/diagnostics.png`, `results/vlbi/uv.png` |

Rendering writes its outputs to `out/`, which is not tracked in git.

## What is computed, and how

**Kerr ray tracer** (`src/kerr/tracer.py`)
- Null geodesics run backwards in time from the camera. They use the Hamiltonian in Mino time, with E = 1 and conserved b and q, in Boyer–Lindquist coordinates.
- The integrator is adaptive Dormand–Prince 5(4), with per-step caps on every coordinate. It runs as an NVIDIA Warp GPU kernel in FP32. A 1080p frame (2 M rays) traces in about 12 ms.
- Two numerical points mattered:
  - **Null-constraint projection.** For a distant camera in FP32, the impact parameter is encoded in p_r ≈ −1 only at the 10⁻⁵ level. Without re-projecting p_r onto H = 0, b_crit came out 0.4% low.
  - **True time reversal.** The right-hand side is negated, rather than flipping the spatial momentum. In Kerr, flipping the momentum is equivalent to flipping the spin.
- Equatorial crossings are located by cubic Hermite interpolation. Up to six crossings are recorded per ray, so one trace can be re-shaded with any equatorial emission model.

**Emission** (`src/kerr/emission.py`, `src/kerr/thin.py`)
- Novikov–Thorne/Page–Thorne thin-disk flux, from a numerical integral. Emitters follow Keplerian orbits outside the ISCO and Cunningham plunging orbits inside it.
- The redshift g = ν_obs/ν_emit is computed per crossing. A blackbody seen with redshift g is a blackbody at gT. Its colors come from the CIE 1931 color-matching functions (Wyman et al. 2013 fit).
- The M87*-like 1.3 mm image is optically thin equatorial emission. It uses Johnson-SU profiles (Gralla, Lupsasca & Marrone 2020), and each crossing adds g³J(r). The n = 0, 1, 2 photon-ring images carry 89%, 10% and 0.7% of the flux.

**Simulated Event Horizon Telescope** (`src/vlbi/`)
- 2017 station sites and approximate SEFDs.
- Earth-rotation synthesis for M87, using Thompson, Moran & Swenson eq. 4.1 and a 10° elevation limit. The South Pole Telescope correctly never sees M87.
- Radiometer noise plus a 1% systematic term.
- Random atmospheric phases per station per scan, and amplitude gain errors.
- Imaging uses only closure phases (1,601) and log closure amplitudes (2,306), which are immune to the station errors. The imager is regularized maximum likelihood (entropy, TSV, flux), solved with L-BFGS in rounds that restart from a blurred image. Its normalized cross-correlation with the truth at 20 µas is 0.878.

**Narration**
- Kokoro-82M text-to-speech, voice `af_heart`.
- All 28 segments were transcribed back with Whisper large-v3-turbo. The only differences were number formatting. Pronunciation overrides were needed for "Messier", "microarcseconds" and "Fourier".

**Music** (`src/audio/music.py`)
- Additive-synthesis pads, a drone, plucked arpeggios and a convolution reverb, all synthesized in numpy.
- The harmony stays in D minor and resolves to D major on "a ring appears". The music is ducked under the narration.

**Animation** (`src/film/`)
- skia-python vector overlays composited on the GPU.
- Earth coastlines come from Natural Earth (public domain).

## Physics validation (`tests/test_kerr_validation.py`)

| Test | Result |
|---|---|
| Schwarzschild critical impact parameter (exact √27 M = 5.196152) | 5.196152, error 2.6 × 10⁻⁸ |
| Kerr shadow a = 0.99, edge-on, vs Bardeen (1973) | max edge deviation 0.0099 M (pixel = 0.015 M) |
| Kerr shadow a = 0.9, i = 17°, vs Bardeen | max edge deviation 0.0092 M |
| FP32 GPU vs FP64 reference (autograd of the textbook inverse metric, independent equations) | 100% agreement on capture/escape; median direction error 7 × 10⁻⁴°; 99th percentile 0.013° |

The 1% tail of direction error is FP32 round-off on rays that wind near the photon orbit. Halving the step caps does not reduce it. All of these rays lie inside the photon ring, where the sky is demagnified a thousandfold.

## Honest caveats
- **This is a simulation, not EHT data.** The reconstruction comes from *simulated* observations of a *model* image. The film says so, and the credits point to the real 2019/2022 EHT images.
- **Artistic choices** (all physically motivated):
  - the disk temperature scale (T_max 5,200–7,000 K, chosen for visible color);
  - the procedural disk turbulence (advected by Keplerian rotation, flow-mapped so the shear stays bounded);
  - the procedural sky and background galaxy;
  - color grading, bloom and tone mapping.
- **Spin-up shot:** fluxes are normalized to a non-spinning disk at the same accretion rate, so the disk brightens as spin increases, as it physically should. Other shots normalize each disk to its own peak.
- **Simplified M87\* model:** an equatorial, optically thin flow. It is not a GRMHD simulation. The spin (0.94) and the 17° inclination are illustrative.
- **Approximate data:** SEFDs and station coordinates are approximate, and the observation is a single 24 h track with 6-minute scans.

## Reproduce

```bash
python3 -m venv .venv && echo /path/to/site-packages/with/torch > .venv/lib/python3.12/site-packages/torch.pth
.venv/bin/pip install warp-lang skia-python soundfile "kokoro>=0.9.4" "misaki[en]" pyloudnorm
.venv/bin/python tests/test_kerr_validation.py        # physics checks
.venv/bin/python src/vlbi/simulate.py                 # EHT simulation + imaging -> out/vlbi/sim.npz
.venv/bin/python src/audio/narration.py               # TTS + Whisper verification
.venv/bin/python -m src.film.make info                # timeline
.venv/bin/python -m src.film.make shot open lensing rays disk doppler spin scale eht fourier recon finale credits
.venv/bin/python -m src.film.make assemble            # crossfades -> out/film_video.mp4
.venv/bin/python src/audio/music.py && .venv/bin/python src/audio/mix.py
./tools/finalize.sh                                   # mux video + audio + subtitles
```

Preview any moment with `python -m src.film.make stills <shot> t1 t2 …`. It writes PNGs and a contact sheet to `out/preview/`.
