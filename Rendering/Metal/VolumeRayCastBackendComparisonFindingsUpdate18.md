# Metal vs OpenGL volume ray cast: proxy path is deterministic and matches GL's pre-flip geometry; the i=144 residual is GL's own frame-to-frame drift, not a Metal-vs-GL difference (update 18)

**Date:** 2026-08-08
**Scope:** Re-run the worst-pixel comparison of the NEAREST no-jitter camera-inside test (`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`, 6 frames) with Metal **forced onto the OpenGL-parity proxy-geometry path** — Update 17's Step 1 — and answer whether the frame 3→4 geometry change is caused by the playback `MouseWheelForwardEvent` (it is not).
**Follows:** [Update 17](VolumeRayCastBackendComparisonFindingsUpdate17.md), which root-caused the i=144 residual to a ~1e-4 rad ray-direction tilt from a float32 near-plane ray-origin difference between Metal's *fullscreen* reconstruction and GL's *proxy-mesh* interpolation, and laid out Step 1 (route camera-inside through the proxy path) as the decisive fix.
**Persisted tool:** `BackendComparisonTools/compare_gl_metal_accum.py` (unchanged). Captured logs: `/tmp/bc/update19/gl.log` (GL) and `/tmp/bc/update19/metal_proxy.log` (Metal with `VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0`); fullscreen-path contrast log `/tmp/bc/update18/metal.log` from Update 17.

---

## 1. Conclusion

1. **The `MouseWheelForwardEvent` is NOT the cause of the GL movement.** The camera is read live from `ren->GetActiveCamera()` at render time in both backends and is bit-identical across all 6 frames — and it is the **post-Dolly** camera. Starting from the test's `SetPosition(102.4, 102.4, 60)` toward focal `(100.8, 100.8, 69)`, `Dolly(pow(1.1, 2.0))` (`OnMouseWheelForward` with default `MotionFactor=10`, `MouseWheelMotionFactor=1`) yields `(102.12231405, 102.12231405, 61.56198347)`, which matches GL's logged `cam=(102.122314, 102.122314, 61.5619835)` and Metal's `METAL_CAM` to the last digit. **Both backends apply the wheel event identically**; the dolly happened before the logged frames (during `recorder->Play()`), so every logged frame is post-wheel with the same camera.
2. **GL alone is internally inconsistent across frames.** GL's logged ray step/origin *and* the i=144 sample both change between frame 3 and frame 4 (`step=(-3.90124915e-4, -5.79992993e-5, 1.86620152e-3)` → `(-3.90213419e-4, -5.81052263e-5, 1.86615495e-3)`; i=144 raw `0.0154269` → `0.0181582`) **while the camera, near-plane, sample distance, and matrices stay constant**. The fullscreen Metal path shows the same frame 3→4 boundary (`rayDir` changes `(-0.204495, -0.030411, 0.978395)` → `(-0.204502, -0.030345, 0.978396)`).
3. **The Metal PROXY path is perfectly deterministic.** All 12 SAMPLE groups for the gated pixel `(372,131)` (236 samples each, 6 frames × 2 passes) are byte-identical; the MARCH `rayDir` is `(-0.204523, -0.030396, 0.978390)` in every frame; i=144 is `raw=0.015427` everywhere.
4. **The proxy path matches GL's own pre-flip geometry exactly.** GL frames 1–3 at i=144 give `raw=0.0154269`, `pos=(0.449416, 0.498053, 0.719508)`, `op=0.0179872` — identical to Metal proxy's `raw=0.015427`, `eval=(0.449416, 0.498054, 0.719509)`, `op=0.017987`. Only GL frames 4–6 flip to `raw=0.0181582`, `op=0.400904`.
5. **The residual reported by `compare_gl_metal_accum.py` is a frame-selection artifact.** The tool keeps the **last** occurrence per index, so it compares GL's flipped frame 6 against Metal's constant value. The divergence is GL's own frame 3→4 drift, not a stable Metal-vs-GL difference. When Metal is on the proxy path, it agrees with GL to the last printed digit on 174 of 175 shared samples, and matches GL's pre-flip value on the 175th.

Net: Update 17's Step 1 (proxy path) achieved its goal — Metal is now deterministic and matches GL's geometry; the *remaining* "gap" lives entirely inside GL's frame-to-frame non-determinism at a single TF knot, not in the backends' mutual agreement.

---

## 2. Method

### 2.1 Capture (this update)

Same test and same dump gates as Update 17, with one new piece: the Metal run forces the proxy-geometry camera-inside path via a new debug env override, `VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0` (see section 8.3). The logs:

- `/tmp/bc/update19/gl.log` — OpenGL, `VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=372,380`, 6 frames.
- `/tmp/bc/update19/metal_proxy.log` — Metal, `MTL_LOG_*` shader logging, 6 frames, proxy path.
- `/tmp/bc/update18/metal.log` — Metal, same test, **fullscreen** path (Update 17), used here as the contrast.

The gated pixel pairing is unchanged: Metal `(372,131)` (screenPos, top-left) == GL `(372,380)` (`glReadPixels`, bottom-left, y-flip 511-131).

### 2.2 Analysis

```sh
python3 BackendComparisonTools/compare_gl_metal_accum.py /tmp/bc/update19/gl.log /tmp/bc/update19/metal_proxy.log
```

plus the per-frame greps in section 6/8 (camera constancy, per-frame i=144, per-frame step/origin, MARCH rayDir, SAMPLE-group identity).

---

## 3. The wheel event is applied identically by both backends

The playback stream in the test file (lines 28–32) is:

```
# StreamVersion 1
EnterEvent 298 27 0 0 0 0 0
MouseWheelForwardEvent 200 142 0 0 0 0 0
LeaveEvent 311 71 0 0 0 0 0
```

`MouseWheelForwardEvent` reaches `vtkInteractorStyleTrackballCamera::OnMouseWheelForward`
(`Interaction/Style/vtkInteractorStyleTrackballCamera.cxx`), which does
`Dolly(pow(1.1, factor))` with `factor = MotionFactor * 0.2 * MouseWheelMotionFactor`
= `10 * 0.2 * 1` = `2.0`, i.e. `Dolly(1.21)`.

Starting camera: `(102.4, 102.4, 60)` (test line 109), focal `(100.8, 100.8, 69)`
(the volume-center focal set by `ResetCamera()`; Dolly does not move the focal point).

```
p  = (102.4, 102.4, 60),   fp = (100.8, 100.8, 69)
d  = fp - p = (-1.6, -1.6, 9),   |d| = 9.2802
new |d| = 9.2802 / 1.21 = 7.6691
dir = d/|d| = (-0.17241, -0.17241, 0.96982)
p'  = fp - dir*7.6691 = (102.12231405, 102.12231405, 61.56198347)
```

This is exactly what both backends log in all 6 frames:

- GL `GL_RAY px=(372,380)`: `cam=(102.122314, 102.122314, 61.5619835)` — all 6 frames.
- Metal `METAL_CAM`: `position=(102.122, 102.122, 61.562) focal=(100.8, 100.8, 69) up=(0, 1, 0) viewAngle=30 clipRange=(0.192448, 192.448)` — all 6 frames.

The wheel event therefore fires during `recorder->Play()` (before the 6 logged renders, which happen in `vtkTestingInteractor::Start()` → `vtkTesting::RegressionTest`), and **both backends move the camera by the same 21% dolly**. Metal is not failing to apply it.

---

## 4. Camera / near-plane / sample-distance / matrix constancy (all 6 frames)

| quantity | value (all frames) | backend |
|---|---|---|
| camera position | `(102.122314, 102.122314, 61.5619835)` | GL `GL_RAY cam=`; Metal `METAL_CAM` |
| near-plane | `origin=(-4.28864881, -4.28864881, 24.1236496) normal=(-0.172412191, -0.172412191, 0.969818577)` | GL `GL_NEARPLANE` (9492 lines, all the same plane at two print precisions: 78 high-precision + 9414 lower-precision, identical values) |
| near-plane (volume space) | `volumePos=(0.506559, 0.506559, 0.446101)` | Metal `METAL_NEARPLANE` (6 lines, all identical) |
| sample distance | `0.270058721` | GL `GL_UNIFORMS` (6 lines, all identical); Metal `MTL_OPTABLE sampleDist=0.270059` |
| auto-adjust | `autoAdjust=1 lock=0 sampleDistance=1` | GL `GL_SAMPLING` (6 frames, all identical) |
| volume data | `dt=5 dims=512x512x512 range=(0,4370)` | `TEST_RESAMPLE` (both backends) |

So the scene is constant across all 6 frames in both backends; nothing about the input changes at frame 3→4.

---

## 5. The frame 3→4 geometry change is real in GL and fullscreen-Metal, absent in proxy-Metal

### 5.1 GL: step/origin change at frame 3→4 (camera constant)

```
frames 1-3:  GL_RAY px=(372, 380) cam=(102.122314, ...) origin=(0.505591691, 0.5064044, 0.450773209) step=(-0.000390124915, -5.79992993e-05, 0.00186620152) vpos=(102.00827, 102.105347, 61.9353638)
frames 4-6:  GL_RAY px=(372, 380) cam=(102.122314, ...) origin=(0.505591452, 0.506404161, 0.45077306)   step=(-0.000390213419, -5.81052263e-05, 0.00186615495) vpos=(102.008247, 102.105316, 61.9353447)
```

Camera identical, near-plane identical, yet the interpolated proxy `origin`/`step`/`vpos` shift by ~`2-9e-7` (volume units) at the frame 3→4 boundary.

### 5.2 Metal fullscreen: rayDir change at frame 3→4

```
frames 1-3:  MARCH px=(372,131) rayDir=(-0.204495, -0.030411, 0.978395) tStart=0.002765 tEnd=0.563364
frames 4-6:  MARCH px=(372,131) rayDir=(-0.204502, -0.030345, 0.978396) tStart=0.002766 tEnd=0.563364
```

### 5.3 Metal proxy: fully constant

```
all frames:  MARCH px=(372,131) rayDir=(-0.204523, -0.030396, 0.978390) tStart=0.002765 tEnd=0.563368/0.563367
```

(12 MARCH lines = 6 frames × 2 passes; two lines print `rayDir=-0.204522`, one float32 ulp lower, all else identical.)

---

## 6. Metal proxy is deterministic; GL flips at i=144

### 6.1 SAMPLE-group identity

`metal_proxy.log` contains 12 groups of 236 `SAMPLE px=(372,131)` lines (6 frames × 2 passes, each pass a full march). Every group is byte-identical to every other:

```python
# all 12 groups identical -> proxy path is deterministic frame-to-frame and pass-to-pass
groups[0] == groups[1] == groups[2] == ... == groups[11]   # True
```

### 6.2 i=144 per frame

```
GL frames 1-3:  raw=0.0154269 pos=(0.449416, 0.498053, 0.719508) color=(0.0179872, 0.00948125, 0.00598133) op=0.0179872
GL frames 4-6:  raw=0.0181582 pos=(0.449402, 0.498036, 0.7195)   color=(0.400904, 0.400904, 0.360814)      op=0.400904
Metal proxy:    raw=0.015427  eval=(0.449416, 0.498054, 0.719509) op=0.017987 (identical in all 12 passes)
```

GL frames 1–3 == Metal proxy (to the last printed digit). GL frames 4–6 are the flip. Neighboring samples i=143 and i=145 match exactly between backends in both frame groups.

### 6.3 Why the compare tool still reports a divergence

`compare_gl_metal_accum.py` stores the **last** occurrence per sample index (see `parse_gl`/`parse_metal`: `out[i] = dict(...)`, overwriting per frame). With 6 frames it therefore compares GL's last frame (the flipped frame 6, `raw=0.0181582`) against Metal's constant value (`raw=0.0154270`), reproducing the Update 16/17 headline numbers:

```
   i     raw_GL    raw_MT     op_GL     op_MT
 144 0.0181582 0.0154270 0.400904 0.017987
```

That is the full extent of the remaining gap: one sample, present only because GL itself drifts between frame 3 and 4. (Compare to GL frames 1–3: agreement on all samples, including i=144.)

---

## 7. Interpretation

- The wheel event is a red herring for the frame 3→4 change: both backends carry the same post-Dolly camera in all frames, so no camera move distinguishes frame 3 from frame 4.
- Something in GL's per-render proxy setup produces two slightly different ray geometries across the 6 renders (camera and near-plane constant). The fullscreen Metal path reproduces the same two-geometry pattern, confirming it is a property of the *renderer's proxy-clip / ray-reconstruction arithmetic* across repeated renders, not of a backend mismatch.
- The proxy Metal path removes both the frame-to-frame variation and the backend disagreement: it is constant and matches GL's frames 1–3. Update 17 Step 1 is therefore successful — Metal on the proxy path is the parity reference, and the residual is now attributable to GL's internal non-determinism at the scalar-1150 opacity knot.

---

## 8. Reproduction

### 8.1 Build

```sh
./macos_metal_build.sh --resume --tests   # macOS build with tests (shader logging enabled)
```

### 8.2 Capture (this update's logs)

```sh
mkdir -p /tmp/bc/update19
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('/tmp/bc/dummy_baseline.png')"

# GL (proxy path, 6 frames) at GL pixel (372,380)
VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=372,380 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/update19/gl.log

# Metal FORCED to the proxy path at Metal pixel (372,131)
VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0 \
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/update19/metal_proxy.log
```

Sanity checks:

```sh
rg -c 'GL_SAMPLING autoAdjust'  /tmp/bc/update19/gl.log          # 6 (6 frames)
rg -c 'MARCH px=\(372, 131\)'   /tmp/bc/update19/metal_proxy.log # 12 (6 frames x 2 passes)
rg -c 'METAL_CAM'               /tmp/bc/update19/metal_proxy.log # 6
```

To capture the fullscreen contrast log (Update 17's `/tmp/bc/update18/metal.log`), repeat the Metal command **without** `VTK_METAL_FULLSCREEN_CAMERA_INSIDE` (or with any value other than `"0"`).

### 8.3 The debug override

`VTK_METAL_FULLSCREEN_CAMERA_INSIDE` is a test-only env override added to `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`:

- unset (or any value except `"0"`) → member default `UseFullscreenCameraInside` (this test renders the fullscreen pass, as in Update 17);
- `"0"` → force `cameraInside = false` at the buffer-setup/ray-path decision points, routing the camera-inside render through the OpenGL-parity proxy geometry (`fragment_volume_main`, near-plane-clipped densified mesh).

Implementation: `vtkMetalForceFullscreenCameraInside(bool)` reads `getenv("VTK_METAL_FULLSCREEN_CAMERA_INSIDE")` and is applied at the Phase-6 buffer-setup skip (`SetupBuffers`) and the `cameraInside` decision in the render path. It is a debug/experimental knob, not a product toggle.

### 8.4 Compare

```sh
python3 Rendering/Metal/BackendComparisonTools/compare_gl_metal_accum.py \
  /tmp/bc/update19/gl.log /tmp/bc/update19/metal_proxy.log
```

Expected output (verbatim, this update):

```
GL 175 samples at (372, 380); Metal 236 samples at (372, 131)

=== per-sample region i=130..156 (first divergence i=144) ===
   i     raw_GL    raw_MT     op_GL     op_MT      rgb_GL            rgb_MT
 ...
 144 0.0181582 0.0154270 0.400904 0.017987 (1.00000,1.00000,0.90000) (1.00000,0.52711,0.33253) <--

=== accumulation replay (first divergence + intervals) ===
   i     GL accC            MT accC            dAccC          GL aA   MT aA
 144 (0.629819,0.425132,0.321213) (0.431950,0.222867,0.137856) (+0.1979,+0.2023,+0.1834)  0.6904 0.4926
```

### 8.5 Per-frame greps (the frame 3→4 story)

```sh
# GL camera constancy across frames (must be one unique value):
grep 'GL_RAY px=(372, 380)' /tmp/bc/update19/gl.log | sed 's/.*cam=//' | awk '{print $1}' | sort -u

# GL step per frame (changes between frame 3 and 4):
grep 'GL_RAY px=(372, 380)' /tmp/bc/update19/gl.log | sed 's/.*step=//'

# GL i=144 per frame (flips between frame 3 and 4):
grep 'GL_SAMPLE px=(372, 380) i=144 ' /tmp/bc/update19/gl.log

# Metal proxy rayDir per frame (must be constant):
grep 'MARCH px=(372, 131)' /tmp/bc/update19/metal_proxy.log | sed 's/.*rayDir=/rayDir=/'

# Metal proxy i=144 per frame (must be constant):
grep 'SAMPLE px=(372, 131) i=144 ' /tmp/bc/update19/metal_proxy.log

# Metal proxy SAMPLE-group identity (all 12 groups byte-identical):
python3 - <<'EOF'
import re
groups = []
cur = None
for ln, l in enumerate(open('/tmp/bc/update19/metal_proxy.log'), 1):
    if 'SAMPLE px=(372, 131) i=0 ' in l:
        if cur: groups.append(cur)
        cur = [l]
    elif 'SAMPLE px=(372, 131) ' in l and cur is not None:
        cur.append(l)
if cur: groups.append(cur)
print('groups:', len(groups), [len(g) for g in groups])
print('all identical:', all(groups[0] == g for g in groups))
EOF
```

### 8.6 Dolly math check

```sh
python3 - <<'EOF'
import numpy as np
p = np.array([102.4, 102.4, 60.0]); fp = np.array([100.8, 100.8, 69.0])
d = fp - p; dist = np.linalg.norm(d)
p2 = fp - (d / dist) * (dist / 1.21)
print(p2)  # [102.12231405 102.12231405  61.56198347] == logged camera
EOF
```

---

## 9. Status and next steps

- Update 16: residual = one flipped TF sample i=144, caused by linear sample drift ~0.03 texel.
- Update 17: drift = tiny ray-direction tilt from a float32 near-plane ray-origin difference between Metal's fullscreen reconstruction and GL's proxy interpolation; planned Step 1 = route camera-inside through the proxy path.
- **Update 18 (this):** the proxy path works — Metal is deterministic and matches GL frames 1–3 exactly. The last-frame compare divergence (i=144) is entirely GL's own frame 3→4 drift (camera/near-plane/sample-distance constant), which the fullscreen path also exhibits. The wheel event is proven irrelevant (identical post-Dolly camera in both backends).

Remaining open question: why does GL (and fullscreen-Metal) produce two slightly different ray geometries across repeated renders with a constant camera and near-plane? Options:

1. **Accept.** The residual exists only because GL's own renders disagree at one scalar-1150-knot sample; Metal-on-proxy already agrees with GL's pre-flip frames. At ~1e-7 volume-unit step/origin difference it only flips on near-step data boundaries.
2. **Investigate GL's per-render proxy rebuild** (what differs between render 3 and 4 in `vtkOpenGLGPUVolumeRayCastMapper`'s `ClipConvexPolyData`/densify path given identical camera+plane inputs) — this is a GL-internal determinism issue, not a Metal gap.
3. **Make the comparison frame-selection explicit** (compare first frame, or make GL deterministic) if the tool's last-frame behavior is deemed misleading.

The debug override (`VTK_METAL_FULLSCREEN_CAMERA_INSIDE`), the dump hooks, and this report are the deliverables of this update.
