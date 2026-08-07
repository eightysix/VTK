#!/usr/bin/env python3
"""Pixel-level Metal-vs-GL delta profile for two backend renders.

Beyond analyze.py's global stats, prints: signed delta at specific pixels,
max-|delta| location, the outer 8-px border band vs interior masked counts, a
luminance-binned breakdown, and a 16x16 block map of mean|delta|.

Usage: python3 image_delta_profile.py <GL.png> <Metal.png> [pixels...]
Example: python3 image_delta_profile.py co_OpenGL.png co_Metal.png 0,256 113,45 256,510
"""
import sys
from PIL import Image
import numpy as np

gl = np.array(Image.open(sys.argv[1])).astype(float)
mt = np.array(Image.open(sys.argv[2])).astype(float)
pixels = [tuple(int(v) for v in p.split(',')) for p in sys.argv[3:]]

d = mt - gl
md = np.abs(d).max(axis=2)
mask = md >= 5

for p in pixels:
    i, j = p[1], p[0]
    print(f'px {p}: GL {gl[i, j].round(0).astype(int)} '
          f'Metal {mt[i, j].round(0).astype(int)} d {d[i, j].round(0).astype(int)}')

i, j = np.unravel_index(np.argmax(md), md.shape)
print(f'max d at ({j},{i}): GL {gl[i, j].round(0).astype(int)} '
      f'Metal {mt[i, j].round(0).astype(int)}')

w = 8
border = np.zeros_like(mask)
border[:w, :] = border[-w:, :] = border[:, :w] = border[:, -w:] = True
print(f'masked total {mask.sum()}  border-band {(mask & border).sum()} '
      f'interior {(mask & ~border).sum()}  mean|d| border {md[border].mean():.2f}')

lum = gl.mean(axis=2)
print('luminance bins: lo hi n mean|d| maskedfrac')
for lo, hi in [(0, 30), (30, 80), (80, 150), (150, 220), (220, 256)]:
    m = (lum >= lo) & (lum < hi)
    if m.sum():
        print(f'  {lo:3d} {hi:3d} {m.sum():6d} {md[m].mean():5.2f} '
              f'{((md >= 5)[m]).mean():.2f}')

print('block mean|d| (16x16)')
B, G = 32, 16
for r in range(G):
    print(' '.join(f'{md[r*B:(r+1)*B, c*B:(c+1)*B].mean():4.1f}' for c in range(G)))
