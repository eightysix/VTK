#!/usr/bin/env python3
"""Validate the frame-6-aligned GL pre-store gf dump against clean GL
(VolumeRayCastBackendComparisonFindingsUpdate59.md, section 1).

The GL float dump is a re-render into a black RGBA32F FBO with blend disabled
(DumpCleanGLFloats, VTK_GL_FLOAT_DUMP). It used to run once (frame 1) while the
stored image is frame 6 and the camera animates, so at knife-edge (grid-aligned)
rays the dump disagreed with the image by up to 6+ u8 (update 58 §3). The
update-59 change makes it dump on every frame (overwriting), so the final file
content is frame 6, aligned with the stored image.

Findings it reproduces:
  - GL clean image == round_half_even((gf_dump + 26/255*(1-a))*255) at
    262,141 / 262,144 px.
  - The 3 failing px ((323,225),(203,278),(400,467)) are ultra-boundary
    straddlers (gf sits ~0.4 u8 from the rounding boundary, so a sub-ulp
    re-render jitter flips them), and NONE of them is in the 188-px
    Metal-diff set -> the aligned dump is trustworthy at every pixel the
    Metal comparison cares about.
  - Caveat: the dump-run's own stored image is corrupted (64k px vs clean GL),
    so pair the dump with a separate clean image, never the dump-run image.

Inputs (see README.md in this directory for regeneration):
    BC_DATA/u60_gl_float.raw   frame-6-aligned GL gf dump (float32 RGBA,
                               512x512x4, row 0 = gl_FragCoord y 0)
    BC_DATA/u60_gl_clean.png   clean GL stored image
    BC_DATA/u59_metal.png      Metal stored image (for the 188-px overlap check)

Usage: BC_DATA=/path/to/data python3 aligned_gl_gf_validate.py
"""
import os
import numpy as np
from PIL import Image

BC = os.environ.get("BC_DATA", "/tmp/bc")
N = 512
BLEND = 26.0 / 255.0  # SetBackground(0.1,0.1,0.1) stored as u8 (update 56)


def rint_half_even(x):
    f = np.floor(x)
    r = x - f
    return np.where(r > 0.5, f + 1, np.where(r < 0.5, f, np.where(f % 2 == 0, f, f + 1)))


def main():
    raw = np.fromfile(os.path.join(BC, 'u60_gl_float.raw'), dtype=np.float32).reshape(N, N, 4)
    gl_gf = raw[::-1, :, :3].astype(np.float64)   # flip rows (OpenGL bottom-origin)
    gl_a = raw[::-1, :, 3].astype(np.float64)

    gl_img = np.array(Image.open(os.path.join(BC, 'u60_gl_clean.png'))).astype(np.float64)
    mt_img = np.array(Image.open(os.path.join(BC, 'u59_metal.png'))).astype(np.float64)

    stored_model = rint_half_even((gl_gf + BLEND * (1.0 - gl_a[:, :, None])) * 255.0)
    ok = (np.abs(stored_model - gl_img) < 0.5).all(axis=2)
    print('GL clean image == model(GL gf dump) px: %d / %d' % (ok.sum(), N * N))
    bad = np.argwhere(~ok)
    print('failing px:', bad.shape[0])
    for y, x in bad[:10]:
        print('  (%d,%d) img=%s model=%s gf=%s a=%.4f' % (
            x, y, gl_img[y, x].astype(int).tolist(), stored_model[y, x].astype(int).tolist(),
            np.round(gl_gf[y, x] * 255, 2).tolist(), gl_a[y, x]))

    dpx = np.all(gl_img == mt_img, axis=2) == False
    overlap = [tuple(int(v) for v in c) for c in bad if dpx[c[0], c[1]]]
    print('GL-self-inconsistent px in the 188 Metal-diff set:', len(overlap), overlap)
    print('(expect 0 -> the aligned dump is trustworthy at all Metal-diff px)')


if __name__ == '__main__':
    main()
