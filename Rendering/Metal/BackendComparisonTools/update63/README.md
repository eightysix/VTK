# Update 63 — window-position f32 rounding REFUTED as the varying-path bias; the residual is localized to a rasterizer-level sample displacement (~+0.03,-0.03 px, GL > Metal) reproducible by no analytic pixel-center model

Persistent copies of the scripts behind
[`VolumeRayCastBackendComparisonFindingsUpdate63.md`](../../VolumeRayCastBackendComparisonFindingsUpdate63.md),
which closes the three open hypotheses of update 62:

- **The varying-path bias is NOT f32 window-position rounding** (`window_u63.py`).
  Five rounding variants of the enormous vertex window coords (exact f64
  `(ndc+1)*256`, sequential f32 mul-add / add-mul, f64-product-then-f32,
  half-framebuffer f32 `ndc*256+256`) give byte-identical window coords and
  identical implied sample-NDC offsets (~1e-7, ~2000× below the ~(+2.5e-4,
  −3.5e-4) NDC offset update 62 needed). The bias is a real sub-pixel
  effective-sample displacement.
- **No single sample offset reproduces GL's texcoord at all pixels**
  (`refine_u63.py`). Per-pixel 3/3-zero-ulp regions at step 1e-6 are broad but
  disjoint ((349,255) needs dx ≤ +1.19e-4, (482,33) needs dx ≥ +3.15e-4); 5 of
  14 pixels have no 3/3 point at all; no (mx%2,my%2) phase correlation.
- **Per-pixel effective weights are internally consistent but displaced in both
  backends** (`backout_u63.py`, `backout_both_u63.py`, `displace_u63.py`). A
  single weight set fits the 3 texcoord channels to ~1e-15 (f64) at every
  pixel; the implied sample displacement is (+2.9e-4,−2.9e-4) NDC for GL and
  (+1.8e-4,−1.7e-4) for Metal (means over 14 px) — the same direction, with
  **GL ~(+0.03,−0.03) px beyond Metal**.
- **The residual GL-vs-Metal delta is localized to the interpolated texcoord**
  (`texcompare_u63.py`): GL `tex=` vs Metal `localPos=` (= the interpolated
  texcoord, `anchorTex = in.texcoord`) differ by −3..+3 ulps per pixel, mostly
  GL 1-2 ulps below Metal — despite bit-identical per-vertex data (update 60)
  and position-path weights (update 61). This propagates into the ray anchor
  and drives the 188-px knife-edge image delta.
- **No pixel-center analytic model reproduces either backend**
  (`model_u63.py`): affine / perspective `1/w` / perspective `rcp(1/w)` /
  inverse-w all give 0/14 pixels with 3/3 channels at 0 ulps on both backends.

Conclusion: the varying (attribute) interpolator on this hardware samples
effectively off pixel center in the same direction in both drivers, GL further
than Metal; that driver-level attribute-interpolation difference (not the
shader math) is the residual floor unless Metal can emulate GL's attribute
interpolation exactly — which pixel-center analytics rule out.

## What is persisted here

| file | role |
|---|---|
| `window_u63.py` | 5 window-coordinate rounding variants; implied sample-NDC offset `Σλᵢ·refᵢ` from the rounded-window barycentric at each pixel center, + texcoord ulps vs GL (update 63 §2) |
| `refine_u63.py` | Fine per-pixel scan (step 1e-6, ±6e-4) of the full 3/3-zero-ulp (dx,dy) region per pixel + pixel phase (update 63 §3) |
| `backout_u63.py` | Least-squares back-out of effective perspective weights from the 3 logged texcoord channels (GL), with fit residual (update 63 §4) |
| `backout_both_u63.py` | Same back-out for GL and Metal side by side; max \|Δλ\| per pixel (update 63 §4) |
| `displace_u63.py` | Converts each backed-out weight set into an effective sample displacement from the pixel center (NDC and px), per backend + GL−Metal (update 63 §4) |
| `texcompare_u63.py` | Direct per-pixel GL `tex=` vs Metal `localPos=` ulp comparison (update 63 §5) |
| `model_u63.py` | Analytic pixel-center models (affine / persp 1/w / persp rcp / inverse-w) 3/3-zero-ulp counts on both backends (update 63 §5) |

The **input data is NOT committed** (large; `u62_gl_vlog.log` 44 MB,
`u62_metal.log` 27 MB). It lives in `/tmp/bc` on the machine where the findings
were produced and can be regenerated with the commands below. Point the scripts
at any copy with `BC_DATA=/path/to/data`.

## Data inventory (not committed — regenerate or keep in `/tmp/bc`)

| file | size | source |
|---|---|---|
| `u62_gl_vlog.log` | 44 MB | debug-injected GL run: `GL_VERT` (94 vids, 11 channels incl. per-vertex texcoord), `GL_RAY` (interpolated clip+tex+flatVid+primId at the gated px) |
| `u62_metal.log` | 27 MB | Metal run: `vertex_volume_main` (94 vids, clip+uvx+texcoord+modelPos) and `DEBUG STEP` (interpolated clip+localPos+screenPos at the gated px) |

Both are frame-6 (last-occurrence-per-key) captures of
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`.

## Regeneration instructions

Identical to updates 61-62 (`BackendComparisonTools/update62/README.md`): build
with `./macos_metal_build.sh --resume --tests`, capture the GL log with
`VTK_GL_RAY_DUMP=1 VTK_GL_VERTEX_DUMP=1` and the Metal log with
`MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=1073741824 MTL_LOG_TO_STDERR=1`,
both against a dummy baseline so the test fails-and-dumps its render. See the
update-61 README for the exact commands.

## Running the scripts

```sh
cd Rendering/Metal/BackendComparisonTools/update63
BC_DATA=/tmp/bc python3 window_u63.py     # -> all 5 rounding variants: identical implied offsets ~1e-7 NDC (refutes f32 window rounding)
BC_DATA=/tmp/bc python3 refine_u63.py     # -> per-px 3/3 regions broad + disjoint; 5 px never 3/3 (4-5 min)
BC_DATA=/tmp/bc python3 backout_u63.py    # -> one weight set fits 3 channels to ~1e-15; displaced ~1e-6 (l2)
BC_DATA=/tmp/bc python3 backout_both_u63.py  # -> GL vs Metal weights differ ~3-7e-7 (GL l0 >, l2 <)
BC_DATA=/tmp/bc python3 displace_u63.py   # -> GL disp mean (+2.9e-4,-2.9e-4) NDC, Metal (+1.8e-4,-1.7e-4); GL-MT ~(+0.03,-0.03) px
BC_DATA=/tmp/bc python3 texcompare_u63.py # -> interpolated texcoord GL vs Metal -3..+3 ulps
BC_DATA=/tmp/bc python3 model_u63.py      # -> no pixel-center analytic model hits 3/3 on either backend (0/14)
```

Requires `numpy`. All scripts read the two log files from `BC_DATA` (default
`/tmp/bc`), parse last-occurrence-per-key (frame 6), and convert GL keys
`(mx, 511-my)` ↔ Metal `(mx, my)` internally; the rasterizer sample NDC is
`((mx+0.5)/256−1, (511−my+0.5)/256−1)` (GL gl_FragCoord y-up convention). The
framebuffer is **512 px wide** → vertex window coords = `(ndc+1)*256`.

## Commit / history anchors

- Update 63 doc: `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate63.md`
- Update 62: doc `e1e312445a`, tooling `c75fc18c1a`; update 61: doc `7919acbb98`, tooling `4c250a654d`; update 60 `8109341d27`; update 59 tooling (`BackendComparisonTools/update59/`).
- Canonical capture-procedure cheatsheet:
  `Rendering/Metal/VolumeRayCastBackendComparisonProcedures.md`.
