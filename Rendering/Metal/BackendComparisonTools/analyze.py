import sys
from PIL import Image
import numpy as np

gl = np.array(Image.open(sys.argv[1])).astype(float)   # OpenGL render
mt = np.array(Image.open(sys.argv[2])).astype(float)   # Metal render
label = sys.argv[3]

d = mt - gl                       # signed per-channel delta
md = np.abs(d).max(axis=2)        # max-channel |delta|
mask = md >= 5                    # pixels that differ by >= 5/255

print(f'=== {label} ===')
print(f'  center GL {gl[256,256].astype(int)} Metal {mt[256,256].astype(int)}')
print(f'  delta mean {d.mean():+.2f}  mean|d| {md.mean():.2f}  max|d| {md.max():.0f}')
for c, ch in zip(range(3), 'RGB'):
    a = np.polyfit(gl[..., c].ravel(), mt[..., c].ravel(), 1)
    print(f'  {ch}: metal = {a[0]:.4f}*gl + {a[1]:.2f}')
print(f'  masked (>=5) px: {mask.sum()}')

heat = np.zeros_like(gl)
heat[mask] = md[mask, None] * 255.0 / max(md.max(), 1e-9)   # scale to [0,255]
Image.fromarray(heat.astype(np.uint8)).save(f'{label}_delta_heatmap.png')
Image.fromarray((mask.astype(np.uint8) * 255)).save(f'{label}_delta_mask.png')
