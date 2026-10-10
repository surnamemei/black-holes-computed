"""Cut the master film into 720p web chapters (2-pass H.264, <= ~13 MB each) + WebVTT subtitles."""
import json, os, re, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.film.timeline import schedule

ROOT = os.path.join(os.path.dirname(__file__), '..')
SRC = os.path.join(ROOT, 'out', 'how_to_photograph_a_black_hole.mp4')
OUT = os.path.join(ROOT, 'web', 'chapters')
CHAPTERS = [("Fifty-five million light-years", ["open"]), ("Gravity bends light", ["lensing", "rays"]),
            ("The glowing disk", ["disk", "doppler", "spin"]), ("Too small to see", ["scale"]),
            ("A telescope the size of Earth", ["eht", "fourier"]), ("A ring appears", ["recon", "finale", "credits"])]

sched, total = schedule()
st = {s.name: (t0, s.duration, xf) for s, t0, xf in sched}
order = [s.name for s, _, _ in sched]
bounds = [0.0]
for title, shots in CHAPTERS[:-1]:
    last = shots[-1]
    nxt = order[order.index(last) + 1]
    end_last = st[last][0] + st[last][1]
    bounds.append(round((st[nxt][0] + end_last) / 2, 3))
bounds.append(total)

def srt_cues(path):
    txt = open(path).read().strip().split('\n\n')
    for block in txt:
        lines = block.split('\n')
        a, b = [x.strip() for x in lines[1].split('-->')]
        f = lambda s: (lambda h, m, rest: int(h) * 3600 + int(m) * 60 + float(rest.replace(',', '.')))(*s.split(':'))
        yield f(a), f(b), '\n'.join(lines[2:])

def vtt_time(t):
    h, r = divmod(max(t, 0), 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

cues = list(srt_cues(os.path.join(ROOT, 'out', 'film.srt')))
meta = []
TARGET_MB = 12.6
for k, (title, shots) in enumerate(CHAPTERS):
    a, b = bounds[k], bounds[k + 1]
    dur = b - a
    v_kbps = 1300  # uniform quality; total stays under one 64 MB publish
    base = os.path.join(OUT, f"ch{k + 1}")
    common = ['-ss', f'{a:.3f}', '-t', f'{dur:.3f}', '-i', SRC, '-vf', 'scale=1280:720:flags=lanczos',
              '-c:v', 'libx264', '-preset', 'slow', '-profile:v', 'high', '-b:v', f'{v_kbps}k',
              '-maxrate', f'{int(v_kbps * 1.6)}k', '-bufsize', f'{int(v_kbps * 3)}k', '-pix_fmt', 'yuv420p']
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + common + ['-pass', '1', '-passlogfile', base, '-an', '-f', 'mp4', '/dev/null'], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + common + ['-pass', '2', '-passlogfile', base,
                    '-af', f'afade=t=in:d=0.03,afade=t=out:st={dur - 0.03:.3f}:d=0.03', '-c:a', 'aac', '-b:a', '112k',
                    '-movflags', '+faststart', base + '.mp4'], check=True)
    with open(base + '.vtt', 'w') as fh:
        fh.write('WEBVTT\n\n')
        for (ca, cb, text) in cues:
            if cb > a and ca < b:
                fh.write(f"{vtt_time(ca - a)} --> {vtt_time(min(cb, b) - a)}\n{text}\n\n")
    size = os.path.getsize(base + '.mp4') / 2 ** 20
    meta.append(dict(n=k + 1, title=title, start=a, duration=round(dur, 2), mb=round(size, 2)))
    print(f"ch{k + 1} {title:32s} {a:7.2f}s +{dur:6.2f}s  {v_kbps} kbps -> {size:5.2f} MB", flush=True)
for f in os.listdir(OUT):
    if f.endswith('.log') or f.endswith('.mbtree'):
        os.remove(os.path.join(OUT, f))
json.dump(meta, open(os.path.join(OUT, 'chapters.json'), 'w'), indent=1)
print('total MB', round(sum(m['mb'] for m in meta), 1))
