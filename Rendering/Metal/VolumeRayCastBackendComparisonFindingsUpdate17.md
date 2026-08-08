# Metal vs OpenGL volume ray cast: the sample-step tilt is a float32 near-plane ray-origin difference; camera and matrices are bit-identical (update 17)

**Date:** 2026-08-08
**Scope:** Full-precision step / ray / camera comparison at the worst residual pixel of the NEAREST no-jitter camera-inside residual, Metal `(372,131)` == GL `(372,380)`, `max|d|=22`, from the `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter` test (6 frames, deterministic per backend). This is the "next step" requested by [Update 16](VolumeRayCastBackendComparisonFindingsUpdate16.md) section 6 (log `g_dirStep` / `evalStep` / ray / camera at `%.9e` and diff them directly).
**Follows:** [Update 16](VolumeRayCastBackendComparisonFindingsUpdate16.md), which fit the per-sample positions and found the step vectors differ per-axis (x +0.03%, y +0.41%, z -0.003%), i.e. a tiny ray-direction tilt, but could not separate ray-direction from step/scale composition (the positions were logged at 7 decimals).
**Persisted tool:** `BackendComparisonTools/compare_gl_metal_steps.py` (sections 1-10 below). Captured logs: `/tmp/bc/update18/metal.log` (Metal) and `/tmp/bc/update18/gl.log` (GL).

---

## 1. Conclusion

The step difference (the Update 16 drift, ~0.03 texel at sample i=144) is **real** and is now fully decomposed. Logging the entire float32 chain at `%.9e` (exact round-trip for float32) proves:

1. **The camera is bit-identical.** Metal `volumeUniforms.cameraVolumePos` and GL `eyePosObjs / boundsSize` are the SAME float32 value at every one of the 6 frames (`|diff| = 0.0e+00`).
2. **The matrices and sample distance are bit-identical.** Replaying Metal's own logged `dirObj` through the GL-logged `invTexDataset*cellToPoint` and `sampleDistanceWorld` reproduces Metal's logged `evalStep` to `3.6e-12 .. 2.9e-11` (pure matrix-chain rounding). The GL↔Metal swap-in test is `6.98e-08 .. 2.73e-07` of the same magnitude.
3. **The step difference is caused entirely by the ray direction.** Section 4 of the tool shows GL's step replayed with Metal's direction equals Metal's step, not GL's step — i.e. `dirObj` (the direction) is the only input that differs.
4. **The ray directions differ because the near-plane ray ORIGINS differ** by `2-5e-7` of the unit volume (`4-11e-5` object units, `~0.03` texels of 512^3), which is `7-17x` above the GL 23-bit decode noise floor of the logged `vpos` (`±3e-8` volume). The direction difference implied by that origin difference, `~3.6e-5 .. 1.25e-4` rad (step-angle and drift estimates bracket), tilts the step and accumulates to the single TF flip at i=144.
5. **This test renders through Metal's FULLSCREEN pass** (`fragment_volume_fullscreen_main`, camera-inside path), so Metal's `localPos` is `s.entryPoint = cameraPos + rayDir*tStart` computed in float32, while GL's `vpos` is the near-plane point interpolated from its densified `ClipConvexPolyData` proxy mesh. The two backends compute "the ray start" by two different-but-equivalent float32 chains; they agree to `2-5e-7` volume and no more.

Net: there is no remaining algorithmic or TF difference. The residual is a float32-level difference in how the two backends derive the near-plane ray through a pixel (fullscreen inverse-projection reconstruction vs proxy-mesh clip interpolation). This is the kind of difference that can only be removed by making one backend reproduce the other's exact arithmetic.

---

## 2. Method

### 2.1 Captures

GL: existing `GL_RAY` per-frame dump (`VTK_GL_RAY_DUMP=1`) extended with `step=`, `sampleDist=`, `invTexDataset` diag, `cellToPoint` diag/offset, `eyePosObjs`, and per-frame 23-bit `vpos` (from `VTK_GL_POSITION_DUMP`), at the y-flipped pixel (372,380).

Metal: `DEBUG MARCH` + `DEBUG STEP` blocks in `marchVolumeUnified` (gated by `debugMarchGate` to the paired pixel), logging at `%0.9e`:

```
MARCH px=(372, 131) camera=(...) rayDir=(...) tStart=... p0o=(...) entry=(...) ...
STEP  px=(372, 131) cameraVol=(..., ..., ...) localPos=(..., ..., ...) rayDir=(..., ..., ...)
      dirObj=(...) evalStep=(...) texStep=(...) boundsSize=(...) sampleDistanceWorld=... ctpScale=(...) ctpOffset=(...)
```

- `rayDir`      : normalized object-space direction (proxy) / reconstructed NDC direction (fullscreen)
- `dirObj`      : `normalize(rayDir * boundsSize)` (GL's object-space normalize)
- `evalStep`    : `(adjustedLin * dirObj) * sampleDistanceWorld` — the per-sample step
- `localPos`    : proxy interpolated position, or `s.entryPoint` on the fullscreen path
- `boundsSize`  : `volumeBoundsMax - volumeBoundsMin`; `sampleDistanceWorld`: GL `in_sampleDistance`

### 2.2 Analysis

`python3 BackendComparisonTools/compare_gl_metal_steps.py /tmp/bc/update18/metal.log /tmp/bc/update18/gl.log`

pairs Metal `(372,131)` with GL `(372,380)` across the 6 frames and prints sections 1-10 (tables below). Every arithmetic step is done in float32 (`np.float32`) and `glsl_normalize` (multiply by `1/length`, as the shaders do). Sections 3 and 4 are self-consistency checks on Metal's own values; sections 5-6 bound GL's print-noise; sections 7-10 are the cross-backend comparisons.

---

## 3. The chain, verified (Metal's own logged values)

```
frame 1:  rayDir=(-2.044947594e-01, -3.041079268e-02, +9.783952236e-01)
          dirObj=(-2.917522788e-01, -4.338702187e-02, +9.555094242e-01)
          evalStep=(-3.900613228e-04, -5.800673898e-05, +1.866229344e-03)
frame 4:  rayDir=(-2.045018375e-01, -3.034548461e-02, +9.783957005e-01)
          dirObj=(-2.917626202e-01, -4.329387844e-02, +9.555104971e-01)
          evalStep=(-3.900751180e-04, -5.788221097e-05, +1.866231556e-03)
```

(The camera and matrices are constant across all 6 frames; the volume rotates between frame 3 and 4, so frame 1-3 and frame 4-6 are two distinct geometries. Within each group the values repeat exactly.)

| Check | frame 1-3 | frame 4-6 | meaning |
|---|---|---|---|
| `dirObj == normalize(rayDir*boundsSize)` | `5.96e-08` | `5.96e-08` | Metal's own values are self-consistent (1 float32 ulp) |
| `(adjustedLin_GL * dirObj) * sampleDist == evalStep` | `3.6e-12` | `2.9e-11` | GL's `invTexDataset*cellToPoint` and `sampleDistanceWorld` are float32-identical to Metal's `adjustedLin` / `sampleDistanceWorld` |
| GL `step` replayed with Metal `dirObj` vs GL `step` | `6.98e-08` | `2.73e-07` | swap-in: with Metal's direction the GL matrices produce Metal's step, NOT GL's step |

The last line is the decisive one: feeding Metal's `dirObj` into the (bit-identical) GL matrices reproduces Metal's `evalStep` to `3.6e-12`, while it misses GL's own `step` by `7e-08 .. 2.7e-07`. **The direction is the only differing input.**

## 4. Why GL's logged step is what it is

Sections 5-6 of the tool bound the GL 23-bit `vpos` decode noise: replaying GL's own chain `(adjustedLin * normalize(vpos-eye)) * sampleDist` reproduces GL's logged step to `8e-08` (noise-dominated), and the two-step normalization identity `normalize(normalize(x/S)*S) == normalize(x)` holds to `6e-08`, so the volume-space re-normalize of `p.rayDir` is NOT the tilt source.

## 5. Cross-backend results

```
8) camera:   Metal cameraVol=(0.5065591, 0.5065591, 0.44610134)
             GL  eyePosObjs/S=(0.5065591, 0.5065591, 0.44610134)   |diff|=0.0e+00   (all 6 frames)

9) origin:   Metal s.entryPoint=(0.50599355, 0.506475,   0.44880706)   |diff|=2.086e-07  (frame 1-3)
             GL   vpos/S     =(0.50599337, 0.5064749,  0.44880697)
             Metal s.entryPoint=(0.50599355, 0.50647515, 0.44880712)  |diff|=5.373e-07  (frame 4-6)
             GL   vpos/S     =(0.50599325, 0.5064748,  0.44880685)

10) rayDir:   Metal p.rayDir=(-0.20449476, -0.030410793, 0.9783952)
              normalize(localPos-camVol)=(-0.20449637, -0.03041151,  0.97839487)   frame 1-3
              Metal p.rayDir=(-0.20450184, -0.030345485, 0.9783957)
              normalize(localPos-camVol)=(-0.20449245, -0.03034627,  0.97839767)   frame 4-6
```

Section 10's `rayDir` vs `normalize(localPos - camVol)` mismatch (`1.8e-6 .. 9.6e-6` rad) is **expected on the fullscreen pass**: there `localPos == s.entryPoint == cameraPos + rayDir*tStart`; float32 rounds `camera + delta` (camera ~0.506, `|delta|` ~0.003, ulp 6e-8) and the subtraction `(camera+delta) - camera` is exact, so the recovered delta inherits up to `ulp(camera)/|delta| ~ 1e-5` rad of tilt, concentrated in the small (x) delta component. It is a logging-space artifact, not a backend difference. Section 9's comparison of `s.entryPoint` vs `vpos/S` is therefore the meaningful one, and it is valid: both quantities are the near-plane ray start.

## 6. Interpretation

Both backends are internally consistent and mutually bit-identical in camera, projection-derived matrices, adjustedLin, and sample distance. The rays they march differ only because each computes the pixel's near-plane ray through a different float32 chain:

- **Metal (fullscreen pass):** `rayDir = reconstructRayDir(NDC)` (inverse view-projection multiply in float32), `entryPoint = cameraPos + rayDir * intersectBox(...)`.
- **OpenGL:** the proxy near-plane polygon is clipped from the densified `ClipConvexPolyData` mesh; `in_vertexPos` is interpolated across it; `rayDir = normalize(vertexPos - eyePosObjs)`.

These mathematically-identical constructions disagree at the `1e-4`-object-unit level (the near-plane point differs by `4-11e-5` object units). A `~1e-4` rad tilt in the ray (dominated by the small y component) is exactly the signature Update 16 fit from the sample drift. Over 236 samples it accumulates to ~0.03 texel — enough to flip the single TF sample (i=144) at the bone-surface near-step.

---

## 7. Reproduction

### 7.1 Logs

```
VTK_GL_RAY_DUMP=1 VTK_GL_POSITION_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=372,380 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/update18/gl.log

MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/update18/metal.log
```

### 7.2 Analysis tool

```
python3 BackendComparisonTools/compare_gl_metal_steps.py /tmp/bc/update18/metal.log /tmp/bc/update18/gl.log
```

---

## 8. Status and options

The camera-inside no-transform residual is root-caused end to end:

- Update 16: residual = one flipped TF sample (i=144), caused by linear sample drift ~0.03 texel.
- Update 17: drift = tiny ray-direction tilt (`~1e-4` rad, y-dominant) caused by a `4-11e-5`-object-unit difference between the backends' near-plane ray origins, with camera, matrices and sample distance all bit-identical.

Options if a residual of ~0.03 texel matters (it changes one sample only on data near-steps):

1. **Accept it.** At 0.03 texel it is far below any sampling/interpolation concern for real data (it only flips on near-step data boundaries).
2. **Make Metal reproduce GL's ray construction** for the fullscreen path (interpolate the near-plane position as GL does, instead of `camera + reconstructRayDir*t`). This is the only path that can make the two backends bit-exact here, and it requires replicating GL's proxy clip arithmetic in Metal.
3. **Align the fullscreen ray reconstruction with the proxy path** (use the same ray in both Metal paths) — internal consistency, but does not close the GL gap.

The GL↔Metal step/ray/camera comparison procedure is now complete; the debug logging blocks in `MetalShaders.metal` and `vtkOpenGLGPUVolumeRayCastMapper.cxx` remain uncommitted (kept for reproduction; see the procedures doc).
