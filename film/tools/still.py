"""Render a single still for look development: python tools/still.py out.png [key=value ...]"""
import math, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from PIL import Image
from src.kerr.tracer import Camera
from src.kerr.render import Renderer, DiskStyle, Look, to_uint8

args = dict(a.split('=') for a in sys.argv[2:])
f = lambda k, d: float(args.get(k, d))
a = f('a', 0.9)
t0 = time.time()
R = Renderer(a=a)
torch.cuda.synchronize(); t1 = time.time()
cam = Camera(r=f('r', 40), theta=math.radians(f('inc', 84)), phi=0.0, fov=f('fov', 40),
             forward=(-1.0, f('pitch', 0.0), f('yaw', 0.0)))
style = DiskStyle(T_max=f('T', 9000), doppler=args.get('doppler', 'full'), texture=f('tex', 1.0), t=f('t', 0.0),
                  r_out=f('rout', 24), brightness=f('bright', 1.0))
look = Look(exposure=f('exp', 1.0), bloom=f('bloom', 0.035), sky_gain=f('sky', 1.0), white=f('white', 6.0), saturation=f('sat', 1.0))
W, H = int(f('W', 1920)), int(f('H', 1080))
img, res = R.render(cam, W, H, ss=int(f('ss', 3)), style=style, look=look)
torch.cuda.synchronize(); t2 = time.time()
Image.fromarray(to_uint8(img)).save(sys.argv[1])
print(f'init {t1-t0:.1f}s  render {t2-t1:.2f}s  -> {sys.argv[1]}  (median steps {res.nsteps.float().median():.0f})')
