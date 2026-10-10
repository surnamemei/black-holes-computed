"""Inline data into the page and stage every published file in web/dist."""
import os, shutil, subprocess
W = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(W, '..')
D = os.path.join(W, 'dist'); os.makedirs(os.path.join(D, 'chapters'), exist_ok=True)
page = open(os.path.join(W, 'src', 'index.src.html')).read().replace('/*DATA*/', open(os.path.join(W, 'data.js')).read())
open(os.path.join(D, 'index.html'), 'w').write(page)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', os.path.join(R, 'out', 'poster_4k.png'), '-vf', 'scale=1600:-1:flags=lanczos', '-q:v', '4', os.path.join(D, 'poster.jpg')], check=True)
shutil.copy(os.path.join(R, 'out', 'validation', 'kerr_shadow_a099_i90.png'), os.path.join(D, 'validation.png'))
for f in sorted(os.listdir(os.path.join(W, 'chapters'))):
    if f.endswith('.mp4'):
        shutil.copy(os.path.join(W, 'chapters', f), os.path.join(D, 'chapters', f))
    elif f.endswith('.vtt'):
        shutil.copy(os.path.join(W, 'chapters', f), os.path.join(D, 'chapters', f[:-4] + '.txt'))
tot = 0
for root, _, files in os.walk(D):
    for f in files: tot += os.path.getsize(os.path.join(root, f))
print('index.html KB', round(os.path.getsize(os.path.join(D, 'index.html')) / 1024, 1), '| dist total MB', round(tot / 2**20, 1))
