# Black Holes, Computed

A short film and three interactive instruments about the two ways we know black holes are real: the first picture and the first sound. Everything is computed from general relativity and public data, and the physics is checked against exact results before it is shown.

**Live site:** https://surnamemei.github.io/black-holes-computed/

| | What it is |
|---|---|
| [How to Photograph a Black Hole](https://surnamemei.github.io/black-holes-computed/photograph/) | A 5½-minute narrated film, a live Kerr ray tracer in the browser (WebGL2), and a build-your-own Event Horizon Telescope (gridding, FFT, Högbom CLEAN) |
| [How to Hear Two Black Holes Collide](https://surnamemei.github.io/black-holes-computed/hear/) | GW150914 recovered from LIGO's raw public strain with a from-scratch matched filter, χ² test and time-slide background |
| [The film, 1080p](https://github.com/surnamemei/black-holes-computed/releases/latest) | Master MP4 (H.264, AAC at −16 LUFS, English subtitles), attached to the GitHub release |

## Results at a glance

**Ray tracer** (single precision, GPU), against exact results:

| Check | Exact | Computed |
|---|---|---|
| Critical impact parameter, non-spinning | √27 M = 5.196152 | 5.196152 |
| Kerr shadow edge, a = 0.99, edge-on (Bardeen 1973) | analytic curve | ≤ 0.0099 M (one pixel = 0.015 M) |
| GPU single vs independent double precision | same fate, same sky | 100% agreement, median 7×10⁻⁴° |
| Browser (WebGL) shadow edges, a = 0 and a = 0.99 | ±5.196 M · −2.252 / 6.983 M | ±5.195 M · −2.246 / 6.983 M |

**Simulated Event Horizon Telescope:**
- Inputs: 2017 station sites, thermal noise, and random atmospheric phases.
- Method: an image solved from closure phases and closure amplitudes only.
- Result: correlation with the truth at 20 µas resolution r = 0.878.

**GW150914:**

| Quantity | This analysis | Published |
|---|---|---|
| Network SNR | 23.9 | 24 |
| Hanford / Livingston SNR | 19.8 / 13.4 | ≈ 20 / ≈ 13 |
| Arrival difference | 7.1 ms, Livingston first | 6.9 ms |
| Loudest noise in 44 days of time slides, after χ² | 7.4 | (FAR < 1 / 203,000 yr in the full search) |

## Repository layout

```
docs/          GitHub Pages site (landing page, photograph/, hear/); built by tools/make_pages.py
film/          the film: Kerr ray tracer, emission models, EHT simulation, animation, narration, music
  tests/       physics validation (python tests/test_kerr_validation.py)
  results/     validation figures and numbers, EHT diagnostics, subtitles, narration ASR check
gw150914/      the LIGO analysis: analysis.py -> products.py -> chisq.py; results/ holds the outputs
tools/         make_pages.py, landing.html
```

Reproduction details are in [film/README.md](film/README.md) and [gw150914/README.md](gw150914/README.md). After changing a page source, rebuild the site with:

```bash
python3 tools/make_pages.py --base https://surnamemei.github.io/black-holes-computed/
```

## Prior work

None of the ingredients here is new, and several have been done by teams with far more resources. The closest prior work, with credit:

**Black-hole imagery**
- **Luminet (1979)** made the first image of a black hole with a thin accretion disk.
- **James, von Tunzelmann, Franklin & Thorne (2015)** described the Kerr ray tracing behind *Interstellar*. See [CQG 32, 065001](https://doi.org/10.1088/0264-9381/32/6/065001).
- **NASA Goddard's visualization** by Jeremy Schnittman (2019) shows Doppler beaming and the photon ring. See [SVS 13326](https://svs.gsfc.nasa.gov/13326/).
- **SpaceEngine** renders Kerr black holes in real time. See [its write-up](https://spaceengine.org/news/blog220705/).
- **Browser ray tracers** already exist:
  - [Gargantua](https://discourse.threejs.org/t/gargantua-a-real-time-kerr-black-hole-ray-tracer-react-three-fiber/92834), a Kerr–Schild Hamiltonian tracer with a Novikov–Thorne disk;
  - [blackhole-ts](https://awesome.ecosyste.ms/projects/github.com%2Frmarchet%2Fblackhole-ts);
  - [openPablo/blackhole](https://github.com/openpablo/blackhole), Schwarzschild only;
  - the tutorial [Rendering a black hole that spins](https://fable.grantkot.com/kerr/tutorial.html), which makes the same point about time reversal in Kerr that this project ran into;
  - [Starless](https://github.com/raellic/starless), a CPU tracer.

**Imaging a black hole**
- **Katie Bouman's talk *How to take a picture of a black hole*** (TEDxBeaconStreet, 2016) told this story before the image existed, under a very similar title. See [TED](https://www.ted.com/talks/katie_bouman_how_to_take_a_picture_of_a_black_hole).
- **The Event Horizon Telescope Collaboration (2019)** published the real results. See [ApJL 875, L1](https://doi.org/10.3847/2041-8213/ab0ec7).
- **eht-imaging** (`ehtim`, [achael/eht-imaging](https://github.com/achael/eht-imaging)) is the research-grade library. Its closure-quantity RML imaging (Chael et al. 2018) is what this project's simulation imitates.
- **[APSYNSIM](https://launchpad.net/apsynsim)** (Martí-Vidal) is the closest interactive teaching tool to the telescope lab here. It is a desktop app with live uv coverage, dirty beam, dirty image and CLEAN.
- **Other simulators** include [VNSIM](https://arxiv.org/abs/1808.06726v1) and MeqSilhouette for EHT-realistic observations.

**GW150914**
- **The discovery paper:** Abbott et al. (2016), [PRL 116, 061102](https://doi.org/10.1103/PhysRevLett.116.061102).
- **Matched filtering in tutorials:** [PyCBC tutorials](https://github.com/gwastro/PyCBC-Tutorials) and the [ligo-cbc binder notebook](https://nbviewer.org/github/ligo-cbc/binder/blob/master/tutorial/3_WaveformMatchedFilter.ipynb) do it in the browser through Binder.
- **A full reproduction:** *Reproducing GW150914* ([arXiv:2010.07244](https://arxiv.org/pdf/2010.07244)).
- **Audio:** LIGO published audio of the chirp in 2016.

**What this repository adds** is integration and checking rather than new science:
- the film and three instruments form one coherent, browser-accessible package, made end to end on a single GPU;
- every physics component is tested against exact results, and the tests are published (`film/tests/`);
- the narration is synthesized speech verified word by word with Whisper;
- the gravitational-wave page shows the glitch problem honestly and how the χ² test solves it.

## Credits and licenses

- **Code:** MIT ([LICENSE](LICENSE)).
- **Media** (the film, renders, audio clips): CC BY 4.0.
- **Gravitational-wave data:** This research has made use of data obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration and KAGRA. GWOSC data are released under CC BY 4.0.
- **Coastlines:** Natural Earth (public domain).
- **Software used:**
  - Kokoro-82M speech synthesis (Apache-2.0);
  - Whisper (MIT);
  - LALSuite for IMRPhenomD waveforms (GPL; used, not redistributed);
  - NVIDIA Warp, PyTorch, skia-python, NumPy, SciPy, FFmpeg;
  - fonts from Google Fonts (OFL).

## How this was made

Written, coded and rendered by Claude (Anthropic) on 10 October 2026, in one session on one RTX 5090, at the repository owner's invitation to "decide what you actually want to do".
