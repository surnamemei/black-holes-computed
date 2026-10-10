"""Contact sheet of look variations: python tools/sheet.py out.png 'k=v,k=v;k=v,...' (shared base args via env BASE)"""
import math, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.kerr.tracer import Camera
from src.kerr.render import Renderer, DiskStyle, Look, to_uint8
base = dict(kv.split('=') for kv in os.environ.get('BASE', '').split(',') if kv)
variants = [dict(kv.split('=') for kv in v.split(',') if kv) for v in sys.argv[2].split(';')]
R = Renderer(a=float(base.get('a', 0.9)))
tiles = []
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16) if os.path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf') else None
for v in variants:
    A = {**base, **v}; f = lambda k, d: float(A.get(k, d))
    R.set_spin(f('a', 0.9))
    cam = Camera(r=f('r', 40), theta=math.radians(f('inc', 84)), fov=f('fov', 40), forward=(-1.0, f('pitch', 0.0), f('yaw', 0.0)))
    style = DiskStyle(T_max=f('T', 9000), doppler=A.get('doppler', 'full'), texture=f('tex', 1.0), t=f('t', 0.0), r_out=f('rout', 24), brightness=f('bright', 1.0))
    look = Look(exposure=f('exp', 1.0), bloom=f('bloom', 0.035), sky_gain=f('sky', 1.0), white=f('white', 6.0))
    img, _ = R.render(cam, int(f('W', 960)), int(f('H', 540)), ss=int(f('ss', 3)), style=style, look=look)
    im = Image.fromarray(to_uint8(img)); d = ImageDraw.Draw(im)
    d.text((8, 6), ' '.join(f'{k}={v}' for k, v in v.items()), fill=(255, 255, 0), font=font)
    tiles.append(np.array(im))
cols = int(os.environ.get('COLS', 2))
rows = (len(tiles) + cols - 1) // cols
h, w = tiles[0].shape[:2]
sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
for i, t in enumerate(tiles):
    sheet[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t
Image.fromarray(sheet).save(sys.argv[1]); print('saved', sys.argv[1])
