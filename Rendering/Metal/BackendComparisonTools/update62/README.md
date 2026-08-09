# Update 62 — plane-equation interpolation ≡ barycentric; the varying (texcoord) path uses a DIFFERENT effective sample than the position (clip) path

Persistent copies of the scripts behind
[`VolumeRayCastBackendComparisonFindingsUpdate62.md`](../../VolumeRayCastBackendComparisonFindingsUpdate62.md)
(commit `e1e312445a`, branch `metal-ios`), which closed the last untested
interpolation-formula class and then showed that the ~1-ulp interpolated-anchor
residual comes from the texcoord (varying) and clip (position) interpolation
paths using **different effective sample weights**:

- **Plane-equation / setup-derivative perspective interpolation is numerically
  identical to the barycentric area-ratio model** (`plane_u62.py`): Cramer and
  base-vertex 2×2 solves, f64 and f32, all land in the same sum\|ulps\| band
  (GL 100-112, Mt 59-69) as the barycentric variants (GL 107, Mt 68). The
  formula class is NOT the differentiator; the systematic +2.5 (GL) / +1.5 (Mt)
  ulp bias lives in the effective weights / sample geometry.
- **The position (clip) path is exactly the analytic pixel-center f32-NDC
  weights.** clip.w is reproduced by `1/affine(1/w)` at the sample to +2.5 (GL)
  / +1.5 (Mt) ulps, and logged clip.x/w, clip.y/w equal the pixel-center NDC to
  ~1e-6 (for (397,110): window (397.50019, 401.49920) vs pixel center
  (397.5, 401.5)).
- **The varying (texcoord) path behaves as if interpolated at a sub-pixel
  sample offset.** `solve_u62.py` scans per pixel for a sample-NDC offset
  (dx,dy) making the f64-perspective + f32-NDC model reproduce GL's logged
  texcoord exactly (0 ulps on all 3 channels). Winning offsets cluster tightly:
  dx ∈ [+1e-4, +3.8e-4], dy ∈ [−9e-4, −8e-5] (~+0.06, −0.09 px) — 9/14 knife
  pixels hit 3/3 channels, the other 5 hit 2/3 channels (1-5 ulps residual on
  one channel). Likely mechanism: f32 rounding of the enormous vertex window
  positions (vid 40 y ≈ 251228, vid 93 x ≈ −77907; f32 ulp ≈ 0.015 px) in the
  rasterizer's edge-function/plane setup.
- **The two paths are decoupled.** The same offset that fixes texcoord breaks
  clip.x by thousands of ulps (+5159 … +18599 at the tex-fixing offsets) while
  clip needs offset ≈ 0 (≤23 ulps at the pixel center). So gl_Position
  interpolation (screen-affine, pixel-center weights) and attribute
  interpolation (biased effective sample) run through different hardware paths.

Conclusion: per-vertex data, formula class, pixel center, and position-path
weights are all exonerated; the residual is a varying-path effective-sample
bias common to both backends, with the remaining GL-vs-Metal ~1-3 ulp delta
attributed to the two drivers' differing position post-processing.

## What is persisted here

| file | role |
|---|---|
| `plane_u62.py` | Plane-equation (Cramer) and base-vertex 2×2 (diag) affine interpolation of `a/w` and `1/w`, f64 and f32, vs the barycentric model: (A) clip.w = 1/affine(1/w), (B) clip.x/y reconstruction from the sample, (C) perspective-correct texcoord (update 62 §2-3) |
| `solve_u62.py` | Per-pixel scan for the sample-NDC offset (dx,dy) making all 3 texcoord channels equal GL's logged values exactly (0 ulps); plus the Result-4 clip.x-breakage check (update 62 §4-5) |

The **input data is NOT committed** (large; `u62_gl_vlog.log` 44 MB,
`u62_metal.log` 27 MB). It lives in `/tmp/bc` on the machine where the findings
were produced and can be regenerated with the commands below. Point the scripts
at any copy with `BC_DATA=/path/to/data`.

## Data inventory (not committed — regenerate or keep in `/tmp/bc`)

| file | size | source |
|---|---|---|
| `u62_gl_vlog.log` | 44 MB | debug-injected GL run: `GL_VERT` (94 vids, 11 channels incl. per-vertex texcoord), `GL_RAY` (interpolated clip+tex+flatVid+primId at the gated px), `GL_CAPINDEX` (tri topology) |
| `u62_metal.log` | 27 MB | Metal run: `vertex_volume_main` (94 vids, clip+uvx+texcoord+modelPos) and `DEBUG STEP` (interpolated clip+localPos+screenPos at the gated px) |

Both are frame-6 (last-occurrence-per-key) captures of
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`.

## Regeneration instructions

Identical to update 61 (`BackendComparisonTools/update61/README.md`): build
with `./macos_metal_build.sh --resume --tests`, capture the GL log with
`VTK_GL_RAY_DUMP=1 VTK_GL_VERTEX_DUMP=1` and the Metal log with
`MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=1073741824 MTL_LOG_TO_STDERR=1`,
both against a dummy baseline so the test fails-and-dumps its render. See the
update-61 README for the exact commands.

## Running the scripts

```sh
cd Rendering/Metal/BackendComparisonTools/update62
BC_DATA=/tmp/bc python3 plane_u62.py   # -> A) clip.w 1/affine(1/w): GL +2.5 / Mt +1.5 ulps (f64)
                                      # -> C) texcoord persp: GL 100-112, Mt 59-69 (== barycentric)
BC_DATA=/tmp/bc python3 solve_u62.py   # -> per-px best (dx,dy): 9/14 px all 3 channels at 0 ulps
                                      # -> Result 4: clip.x @0 ≤23 ulps, @tex-offset +5k..+18k ulps
```

Requires `numpy`. Both scripts read the two log files from `BC_DATA` (default
`/tmp/bc`), parse last-occurrence-per-key (frame 6), and convert GL keys
`(mx, 511-my)` ↔ Metal `(mx, my)` internally; the rasterizer sample NDC is
`((mx+0.5)/256−1, (511−my+0.5)/256−1)` (GL gl_FragCoord y-up convention). The
framebuffer is **512 px wide** → vertex window coords = `(ndc+1)*256`.

## Commit / history anchors

- Update 62 doc: commit `e1e312445a` — `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate62.md`
- Prior milestones: update 61 (`7919acbb98` doc, `4c250a654d` tooling),
  update 60 (`8109341d27`), update 59 tooling
  (`BackendComparisonTools/update59/`), update 58 (`651e3ea4ae`),
  update 57 (`8f991da45b`), update 56 (`fdd7281d07`).
- Canonical capture-procedure cheatsheet:
  `Rendering/Metal/VolumeRayCastBackendComparisonProcedures.md`.
