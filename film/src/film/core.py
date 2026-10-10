"""Film framework: timeline, easing, skia drawing, GPU compositing, ffmpeg I/O."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
import skia
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "out")
W, H, FPS = 1920, 1080, 30
DEV = "cuda"

# ----------------------------------------------------------------------------- palette
BG = (0.012, 0.016, 0.03)
AMBER = (1.0, 0.71, 0.28)
CYAN = (0.33, 0.82, 1.0)
WHITE = (0.95, 0.96, 0.98)
GREY = (0.55, 0.59, 0.66)
RED = (1.0, 0.38, 0.30)
BLUE = (0.45, 0.62, 1.0)

FONT_DIR = "/usr/share/fonts/opentype/inter"
_tf_cache = {}


def typeface(name="Regular"):
    if name not in _tf_cache:
        path = os.path.join(FONT_DIR, f"Inter-{name}.otf")
        if name.startswith("Display"):
            path = os.path.join(FONT_DIR, f"InterDisplay-{name[7:]}.otf")
        _tf_cache[name] = skia.Typeface.MakeFromFile(path)
    return _tf_cache[name]


# ----------------------------------------------------------------------------- easing


def clamp01(x):
    if isinstance(x, torch.Tensor):
        return x.clamp(0.0, 1.0)
    return max(0.0, min(1.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp01(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def ease_in_out(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp01(x)
    return x ** 3


def ramp(t, t0, t1, f=smooth):
    """0 before t0, 1 after t1, eased in between."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return f(clamp01((t - t0) / (t1 - t0)))


def window(t, t0, t1, fade_in=0.5, fade_out=0.5):
    """Fade in at t0, hold, fade out ending at t1."""
    return min(ramp(t, t0, t0 + fade_in), 1 - ramp(t, t1 - fade_out, t1))


def lerp(a, b, x):
    return a + (b - a) * x


def loglerp(a, b, x):
    return math.exp(math.log(a) + (math.log(b) - math.log(a)) * x)


# ----------------------------------------------------------------------------- narration timing

_narr = None


def narration_meta():
    global _narr
    if _narr is None:
        _narr = json.load(open(os.path.join(OUT, "audio", "narration", "narration.json")))
    return _narr


@dataclass
class Cue:
    seg: str
    start: float  # seconds from shot start
    duration: float

    @property
    def end(self):
        return self.start + self.duration

    def word(self, w, nth=0):
        """Shot-relative time at which word `w` (case-insensitive) starts (nth occurrence)."""
        k = 0
        for x in narration_meta()[self.seg]["words"]:
            if x["w"].lower().strip(".,:;!?") == w.lower():
                if k == nth:
                    return self.start + x["t0"]
                k += 1
        raise KeyError(f"word {w!r} not in segment {self.seg}")


def layout_cues(segs, lead=1.0, gap=0.8, tail=1.0, gaps=None):
    """Place narration segments back to back; returns (cues, shot duration)."""
    meta = narration_meta()
    t = lead
    cues = []
    for k, s in enumerate(segs):
        d = meta[s]["duration"]
        cues.append(Cue(s, t, d))
        g = gaps[k] if gaps and k < len(gaps) else gap
        t += d + (g if k < len(segs) - 1 else 0)
    return cues, t + tail


class Shot:
    """Base class. Subclasses set self.duration and self.cues and implement frame(t) -> (3,H,W) GPU tensor in [0,1]."""

    name = "shot"
    duration = 5.0
    cues: list = []

    def setup(self):
        pass

    def frame(self, t: float) -> torch.Tensor:
        raise NotImplementedError

    def cue(self, seg):
        for c in self.cues:
            if c.seg == seg:
                return c
        raise KeyError(seg)


# ----------------------------------------------------------------------------- drawing


def rgba(color, alpha=1.0):
    r, g, b = color
    return skia.Color4f(r, g, b, alpha)


class Canvas:
    """A transparent RGBA overlay drawn with skia (premultiplied)."""

    def __init__(self, w=W, h=H):
        self.w, self.h = w, h
        info = skia.ImageInfo.Make(w, h, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        self.surface = skia.Surface.MakeRaster(info)
        self.c = self.surface.getCanvas()
        self.c.clear(skia.ColorTRANSPARENT)

    # --- primitives
    def _paint(self, color, alpha, width=None, stroke=False, glow=0.0, cap="round"):
        p = skia.Paint(AntiAlias=True)
        p.setColor4f(rgba(color, alpha))
        if stroke:
            p.setStyle(skia.Paint.kStroke_Style)
            p.setStrokeWidth(width or 1.0)
            p.setStrokeCap(skia.Paint.kRound_Cap if cap == "round" else skia.Paint.kButt_Cap)
            p.setStrokeJoin(skia.Paint.kRound_Join)
        if glow > 0:
            p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, glow))
        return p

    def polyline(self, pts, color=WHITE, width=2.0, alpha=1.0, glow=0.0, glow_alpha=0.6, dash=None, closed=False):
        if len(pts) < 2 or alpha <= 0:
            return
        path = skia.Path()
        path.moveTo(*pts[0])
        for p in pts[1:]:
            path.lineTo(*p)
        if closed:
            path.close()
        if glow > 0:
            self.c.drawPath(path, self._paint(color, alpha * glow_alpha, width * 2.5, True, glow))
        p = self._paint(color, alpha, width, True)
        if dash:
            p.setPathEffect(skia.DashPathEffect.Make(list(dash), 0))
        self.c.drawPath(path, p)

    def circle(self, x, y, r, fill=None, stroke=None, width=2.0, alpha=1.0, glow=0.0, dash=None):
        if alpha <= 0:
            return
        if fill is not None:
            if glow > 0:
                self.c.drawCircle(x, y, r, self._paint(fill, alpha * 0.7, glow=glow))
            self.c.drawCircle(x, y, r, self._paint(fill, alpha))
        if stroke is not None:
            if glow > 0:
                self.c.drawCircle(x, y, r, self._paint(stroke, alpha * 0.6, width * 2.5, True, glow))
            p = self._paint(stroke, alpha, width, True)
            if dash:
                p.setPathEffect(skia.DashPathEffect.Make(list(dash), 0))
            self.c.drawCircle(x, y, r, p)

    def rect(self, x, y, w, h, fill=None, stroke=None, width=1.5, alpha=1.0, radius=0.0):
        r = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), radius, radius)
        if fill is not None:
            self.c.drawRRect(r, self._paint(fill, alpha))
        if stroke is not None:
            self.c.drawRRect(r, self._paint(stroke, alpha, width, True))

    def arrow(self, x0, y0, x1, y1, color=WHITE, width=2.5, alpha=1.0, head=14.0, glow=0.0):
        self.polyline([(x0, y0), (x1, y1)], color, width, alpha, glow)
        ang = math.atan2(y1 - y0, x1 - x0)
        for s in (-1, 1):
            a = ang + math.pi + s * 0.45
            self.polyline([(x1, y1), (x1 + head * math.cos(a), y1 + head * math.sin(a))], color, width, alpha)

    def text(self, s, x, y, size=36, weight="Regular", color=WHITE, alpha=1.0, align="left", tracking=0.0,
             shadow=0.0, max_alpha_shadow=0.7, backdrop=0.0):
        """y is the baseline. tracking in em units. backdrop: opacity of a dark pill behind the text."""
        if alpha <= 0 or not s:
            return 0.0
        font = skia.Font(typeface(weight), size)
        font.setSubpixel(True)
        font.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
        widths = font.getWidths(font.textToGlyphs(s))
        trk = tracking * size
        total = sum(widths) + trk * (len(s) - 1)
        if align == "center":
            x -= total / 2
        elif align == "right":
            x -= total
        if backdrop > 0:
            pad = size * 0.35
            self.rect(x - pad, y - size * 0.95, total + 2 * pad, size * 1.3, fill=(0.0, 0.0, 0.0),
                      alpha=backdrop * alpha, radius=size * 0.3)
        if shadow > 0:
            p = self._paint((0, 0, 0), alpha * max_alpha_shadow, glow=shadow)
            self._draw_chars(s, widths, x, y, font, trk, p)
        self._draw_chars(s, widths, x, y, font, trk, self._paint(color, alpha))
        return total

    def _draw_chars(self, s, widths, x, y, font, trk, paint):
        if trk == 0:
            self.c.drawString(s, x, y, font, paint)
            return
        cx = x
        for ch, w in zip(s, widths):
            self.c.drawString(ch, cx, y, font, paint)
            cx += w + trk

    def image(self, arr, x, y, w=None, h=None, alpha=1.0, smooth=True):
        """arr: (h, w, 3|4) uint8 or float [0,1] numpy."""
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
        if arr.shape[2] == 3:
            arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], 2)
        img = skia.Image.fromarray(np.ascontiguousarray(arr), colorType=skia.kRGBA_8888_ColorType)
        w = w or arr.shape[1]
        h = h or arr.shape[0]
        p = skia.Paint(AntiAlias=True)
        p.setAlphaf(alpha)
        samp = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear) if smooth else skia.SamplingOptions()
        self.c.drawImageRect(img, skia.Rect.MakeXYWH(x, y, w, h), samp, p)

    def to_tensor(self):
        """Premultiplied RGBA (4,H,W). Note skia-python's toarray() returns UN-premultiplied
        values even for a premultiplied surface, so we premultiply here."""
        a = self.surface.makeImageSnapshot().toarray()
        t = torch.from_numpy(a).to(DEV).permute(2, 0, 1).float() / 255.0
        t[:3] *= t[3:4]
        return t


def over(base, overlay):
    """Composite a premultiplied RGBA overlay (4,H,W) over an RGB base (3,H,W)."""
    a = overlay[3:4]
    return overlay[:3] + base * (1 - a)


def solid(color=BG):
    return torch.tensor(color, device=DEV).view(3, 1, 1).expand(3, H, W).clone()


def vignette_bg(color=BG, strength=0.5):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=DEV), torch.linspace(-1, 1, W, device=DEV), indexing="ij")
    r2 = (xx ** 2 * (16 / 9) ** 2 + yy ** 2) / (1 + (16 / 9) ** 2)
    base = torch.tensor(color, device=DEV).view(3, 1, 1)
    lift = torch.tensor((0.025, 0.035, 0.06), device=DEV).view(3, 1, 1)
    return base + lift * (1 - r2).clamp(0, 1)[None] ** 2 * strength * 2


# ----------------------------------------------------------------------------- ffmpeg I/O


class VideoWriter:
    def __init__(self, path, fps=FPS, w=W, h=H, crf=10, preset="medium", codec="libx264", pix="yuv444p"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", pix, path],
            stdin=subprocess.PIPE)

    def write(self, frame):
        """frame: (3,H,W) tensor in [0,1] or (H,W,3) uint8 numpy."""
        if isinstance(frame, torch.Tensor):
            frame = (frame.clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


class VideoReader:
    def __init__(self, path, w=W, h=H):
        self.w, self.h = w, h
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)

    def read(self):
        b = self.p.stdout.read(self.w * self.h * 3)
        if len(b) < self.w * self.h * 3:
            return None
        return np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        self.p.stdout.close()
        self.p.wait()


def to_np8(frame):
    return (frame.clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy()
