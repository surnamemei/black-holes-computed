"""Shot order and transitions. Each entry: (shot instance, crossfade into the NEXT shot, seconds)."""
from __future__ import annotations

_TL = None


def build():
    global _TL
    if _TL is not None:
        return _TL
    from .shots_bh import DiskShot, DopplerShot, LensingShot, OpenShot, SpinShot
    disk = DiskShot()
    tl = [
        (OpenShot(), 1.5),
        (LensingShot(), 1.0),
        (disk, 0.0),  # hard cut: Doppler continues the same camera move
        (DopplerShot(t_offset=disk.duration), 1.0),
        (SpinShot(), 1.0),
    ]
    try:
        from . import shots_2d  # noqa: F401  (added as they are written)
        tl = shots_2d.extend(tl)
    except ImportError:
        pass
    _TL = tl
    return tl


def schedule():
    """Absolute start time of every shot (shots overlap by the crossfade)."""
    tl = build()
    out, t = [], 0.0
    for k, (shot, xf) in enumerate(tl):
        out.append((shot, t, xf))
        t += shot.duration - xf
    total = out[-1][1] + out[-1][0].duration
    return out, total
