# The frame 3→4 "drift" is a test-harness artifact: `vtkWindowToImageFilter` perturbs the camera view angle (30 → 30.0000008°) in both backends, and the i=144 residual is GL's ~50× larger interpolation sensitivity to that same perturbation (update 19)

**Date:** 2026-08-08
**Scope:** Confirm or refute Update 18's two load-bearing claims — (1) "the camera is constant across all 6 frames" and (2) "the Metal proxy path is perfectly deterministic" — by adding 9-significant-figure printing to Metal's `METAL_CAM` and re-running the same NEAREST no-jitter camera-inside test (`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`, 6 frames) on the OpenGL-parity proxy path.
**Follows:** [Update 18](VolumeRayCastBackendComparisonFindingsUpdate18.md), which concluded the camera/near-plane were constant, the wheel event was irrelevant, and the residual was "GL's own frame-to-frame drift". That conclusion was an artifact of 6-significant-figure log printing.
**Persisted tool:** `BackendComparisonTools/compare_gl_metal_accum.py` (unchanged). New capture: `/tmp/bc/update20/metal_proxy.log` (Metal, proxy path, `VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0`, `METAL_CAM` now 9 sig figs). Existing contrast logs: `/tmp/bc/update19/gl.log` (GL) and `/tmp/bc/update19/metal_proxy.log`; fullscreen contrast `/tmp/bc/update18/metal.log`.

---

## 1. Conclusion

1. **Update 18's "camera constant" claim was a printing-precision artifact.** The camera view angle is **not** constant: it changes from `30` to `30.0000008` degrees between frame 3 and frame 4, in **both** backends, and this is provably caused by `vtkWindowToImageFilter` inside `vtkTesting::RegressionTest`.
2. **The mechanism is a float32 round-trip.** `vtkWindowToImageFilter` snapshots each camera's view angle into a `float` (`viewAngles[i]`), then re-derives it via `2*atan(tan(viewAngles[i]/2)*mag)` and calls `cam->SetViewAngle(DegreesFromRadians(angle))`. The float32 rounding of `RadiansFromDegrees(30)` (0.5235987755982988 → **0.5235987901687622**) survives the tan/atan round trip and yields exactly **30.000000834826057°** — matching the logged `30.0000008` to every printed digit.
3. **The W2IF perturbation is transient and frame-4-aligned.** W2IF works on a `ShallowCopy` of the camera (vtkWindowToImageFilter.cxx:352-354), renders with the copy (view angle 30.0000008°), then restores and deletes it (lines 557-560). So frames 1-3 render with the original camera (view angle exactly 30.0°, confirmed at 9 sig figs), while frames 4-6 are the three `rtW2if->Update()` renders (vtkTesting.cxx:418/428/446, the failed-baseline path) with the copy (30.0000008°). The pre-W2IF frame count matches the 3/3 split seen in both backends.
4. **Update 18's "Metal proxy is perfectly deterministic" claim was also a printing-precision artifact.** The 9-decimal `STEP` shader log shows proxy-Metal's ray geometry changes at the exact same frame 3→4 boundary as GL (`rayDir` shifts ~1.4e-6, `evalStep` shifts ~2e-9; groups 1-3 identical, groups 4-6 identical, different). The 6-decimal `MARCH`/`SAMPLE` logs only hid it. Even at 6 decimals, a handful of samples (e.g. i=64, 95, 126, 120, 182, 213) shift by 1 ulp in the 6th decimal of `t` in frames 4-6.
5. **The i=144 residual is GL's larger sensitivity to the same perturbation, not a backend mismatch.** GL's per-step change (~1e-7) is ~40-50× larger than proxy-Metal's (~2e-9), so over 144 steps GL's sample drifts ~1.4e-5 (crossing the scalar-1150 opacity knot → `raw` flips 0.0154269 → 0.0181582) while proxy-Metal's drifts only ~3e-7 (stays at `raw=0.015427`). The near-plane clip geometry is *unchanged* (the near plane depends only on camera position/focal, both constant); only the projection matrix shifts (~3e-8 relative from the view-angle delta), and each backend's rasterization interpolation amplifies that differently.
6. **The wheel event remains irrelevant** (Update 18's section 3 stands: identical post-Dolly camera in all logged frames of both backends).

Net: the frame 3→4 flip is **not** GL drift, **not** a Metal-vs-GL difference, and **not** nondeterminism — it is a deterministic response of both backends to a 8.3e-7° view-angle perturbation that the test harness itself (`RegressionTest` → `vtkWindowToImageFilter`) injects into the camera between frames 3 and 4.

---

## 2. Method

### 2.1 Capture (this update)

Same test, same dump gates, same gated pixel pairing as Update 18 (`(372,131)` Metal top-left == `(372,380)` GL bottom-left), with one change: `METAL_CAM` in `vtkMetalGPUVolumeRayCastMapper.mm` now prints `viewAngle` at `std::setprecision(9)` (matching the precision the GL stream already carried), and the build was relinked (`.mm` + link only).

- `/tmp/bc/update20/metal_proxy.log` — Metal, proxy path (`VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0`), 9-sig-fig `METAL_CAM`, 6 frames.
- `/tmp/bc/update19/gl.log` — GL (unchanged, reused).
- `/tmp/bc/update18/metal.log` — Metal fullscreen path (Update 17, reused as contrast).

The Metal shader logs (fragment `MARCH`/`STEP`/`SAMPLE`) arrive on a separate OS-log stream from the CPU `std::cerr` logs (`METAL_CAM`), so CPU/shader line order in the raw file is unreliable (one `METAL_CAM` is even split across two lines by an interleaved `vertex_volume_main` log). All frame-identity conclusions therefore rely on **within-stream** ordering only: the 12 `MARCH`/`STEP`/`SAMPLE` groups (6 frames × 2 passes) and the 6 `GL_SAMPLING`/`GL_RAY` frames.

### 2.2 Frame indexing

- GL frames 1-6 = `GL_SAMPLING` at log lines 3, 4618, 8011, 11404, 14800, 18196.
- Metal frames 1-6 = MARCH/STEP/SAMPLE groups 0-11 (2 passes per frame).
- `GL_CAM` is printed by **two** sites (see 4.2). The 9-sig-fig site (position printed as `102.122314, …`) shows `viewAngle=30` for frames 1-3 and `viewAngle=30.0000008` for frames 4-6 (first `30.0000008` at log line 11412). The 6-sig-fig site (`102.122, …`, 9414 lines) prints `viewAngle=30` in every frame and is simply too coarse to resolve the 8.3e-7° delta.
- `METAL_CAM` boundary: `viewAngle=30` (3 prints) → `viewAngle=30.0000008` (3 prints, including one split across log lines 27-28 by shader-stream interleaving).

---

## 3. Root cause: the W2IF float32 view-angle round-trip

### 3.1 The code

`vtkTesting::RegressionTest(double, ostream&)` (Testing/Rendering/vtkTesting.cxx):

```
399:  vtkNew<vtkWindowToImageFilter> rtW2if;
400:  rtW2if->SetInput(this->RenderWindow);
...
416:  this->RenderWindow->Render();            // a pre-W2IF render (view angle 30)
417:  rtW2if->ReadFrontBufferOff();
418:  rtW2if->Update();                        // W2IF render #1 (view angle 30.0000008)
...
428:    rtW2if->Update();                      // W2IF render #2 (FAILED path)
...
446:    rtW2if->Update();                      // W2IF render #3 (still FAILED path)
```

`vtkWindowToImageFilter::RequestData` (Rendering/Core/vtkWindowToImageFilter.cxx):

```
339:  float* viewAngles = new float[numRenderers];            // <-- float storage
350:  viewAngles[i] = vtkMath::RadiansFromDegrees(cams[i]->GetViewAngle());
352:  cam = cams[i]->NewInstance();
353:  cam->ShallowCopy(cams[i]);                               // camera COPY
354:  aren->SetActiveCamera(cam);                              // copy becomes active
...
454:  double angle = 2.0 * atan(tan(viewAngles[i] / 2.0) * mag);
455:  cam->SetViewAngle(vtkMath::DegreesFromRadians(angle));   // perturbed copy
...
464:  this->Render();                                          // renders with the copy
...
557:    cam = aren->GetActiveCamera();
558:    aren->SetActiveCamera(cams[i]);                        // original restored
559:    cams[i]->UnRegister(this);
560:    cam->Delete();                                         // copy deleted
```

So the original camera is never touched; the W2IF tile render (frames 4-6) uses the copy, whose view angle is re-derived through the float32 round-trip.

### 3.2 The arithmetic (verified)

```
rad      (double) = radians(30.0)        = 0.5235987755982988
rad32    (float)  = f32(rad)             = 0.5235987901687622   (+1.457e-8 from rounding)
half     (float32) = rad32/2             = 0.2617993950843811
tan(half)          = 0.2679492002394105
angle = 2*atan(tan(half)*mag)            = 0.5235987901687622   (round trip exact)
degrees  (double) = 30.000000834826057  ->  prints as 30.0000008
delta              = +8.348e-7 degrees
```

`mag` = `(visVP[3]-visVP[1])/(vp[3]-vp[1])` = 1.0 for a full-window tile (single renderer, tile viewport == renderer viewport).

---

## 4. Both backends' cameras are perturbed identically

### 4.1 Metal `METAL_CAM` (now 9 sig figs)

```
viewAngle=30            (3 prints: log lines 6, 11, 16)     <- frames 1-3 (original camera)
viewAngle=30.0000008    (3 prints: lines 27-28, 73106, 91988) <- frames 4-6 (W2IF copy)
```

At `std::setprecision(9)`, `30.0` prints as `30` and `30.0000008` prints as `30.0000008`, so the split is unambiguous: the Metal camera **is** perturbed, exactly as the GL camera is.

### 4.2 GL `GL_CAM`

GL logs the camera from two print sites in the camera-inside branch (vtkOpenGLGPUVolumeRayCastMapper.cxx:1170 and the nearby 6-sig-fig site):

```
9-sig-fig site:  viewAngle=30          frames 1-3  (39 prints, position=(102.122314, ...))
                 viewAngle=30.0000008  frames 4-6  (first at log line 11412)
6-sig-fig site:  viewAngle=30          ALL frames  (9414 prints; 30.0000008 rounds to "30")
```

The 9-sig-fig site confirms `viewAngle=30` **exactly** pre-W2IF (with `setprecision(9)` active, `30.0` prints as `30` and `30.0000008` prints as `30.0000008`) — it was not a rounding of the perturbed value. The 6-sig-fig site is what misled Update 18 into believing the camera was constant across all six frames.

### 4.3 What the view-angle delta does

The near plane depends only on camera position + focal point (both constant), so the near-plane-clipped proxy **mesh** is byte-identical before/after the boundary (`GL_NEARPLANE` and `METAL_NEARPLANE` are constant across all frames). What changes is the **projection matrix**: `tan(15.0000004°)` differs from `tan(15°)` by ~2.8e-8 relative, so clip-space positions shift ~3e-8 relative, which shifts each backend's per-pixel rasterization interpolation of the fixed mesh.

---

## 5. The frame 3→4 geometry change exists in all three paths

### 5.1 GL proxy (frames 1-3 → frames 4-6)

```
frames 1-3: GL_RAY origin=(0.505591691, 0.5064044, 0.450773209)
            step=(-3.90124915e-4, -5.79992993e-5, 1.86620152e-3)
frames 4-6: GL_RAY origin=(0.505591452, 0.506404161, 0.45077306)
            step=(-3.90213419e-4, -5.81052263e-5, 1.86615495e-3)
per-step delta: ~(8.9e-8, 1.06e-7, 4.7e-8)
```

### 5.2 Metal fullscreen (Update 17's log)

```
frames 1-3: MARCH rayDir=(-0.204495, -0.030411, 0.978395)
frames 4-6: MARCH rayDir=(-0.204502, -0.030345, 0.978396)
```

### 5.3 Metal proxy (this update's `STEP` log, 9 decimals)

Pass A (group 0, 2, 4 -> identical; group 6, 8, 10 -> identical):

```
frames 1-3: localPos=(4.951383173e-01, 5.048617125e-01, 5.007355213e-01)
            rayDir=(-2.045230567e-01, -3.039637022e-02, 9.783896208e-01)
            evalStep=(-3.901130985e-04, -5.797890481e-05, 1.866208040e-03)
frames 4-6: localPos=(4.951383471e-01, 5.048617125e-01, 5.007357597e-01)
            rayDir=(-2.045216858e-01, -3.039624728e-02, 9.783899188e-01)
            evalStep=(-3.901106538e-04, -5.797868289e-05, 1.866209321e-03)
deltas: rayDir ~1.4e-6, localPos ~3e-8, evalStep ~2.4e-9
```

Pass B changes at the same boundary (evalStep delta ~1e-9). Every pass-A and pass-B `STEP` line is identical within the pre-W2IF frames (1-3) and identical within the W2IF frames (4-6), and differs between the two regimes — exactly the GL pattern.

### 5.4 The 6-decimal logs masked it

- `MARCH` (6 decimals): pre-W2IF pass A `rayDir=-0.204523`, post-W2IF pass A `rayDir=-0.204522` — a single last-ulp difference that Update 18 attributed to "one float32 ulp lower, all else identical" and did not connect to the frame boundary.
- `SAMPLE` (6 decimals): bodies are identical across all 12 groups *except* a few samples in frames 4-6 that shift by 1 ulp in the 6th decimal of `t` (i=64, 95, 126 for groups 6-9; i=120, 182, 213 for groups 10-11).
- Update 18's "all 12 groups byte-identical" was true only after stripping timestamps and ignoring those sub-6th-decimal `t` shifts.

---

## 6. Why GL flips at i=144 and proxy-Metal does not

| quantity | GL | Metal proxy |
|---|---|---|
| per-step delta at frame 3→4 | ~1e-7 (origin/step) | ~2e-9 (evalStep) |
| accumulated position drift at i=144 (~144 steps) | ~1.4e-5 | ~3e-7 |
| observed i=144 | `raw 0.0154269` → `0.0181582` (flip; `pos (0.449416,0.498053,0.719508)` → `(0.449402,0.498036,0.7195)`) | `raw 0.015427` constant (below knot) |
| crosses scalar-1150 opacity knot? | yes (frames 4-6) | no |

GL's per-step change is ~40-50× larger than proxy-Metal's (8.9e-8 / 2.4e-9 ≈ 37×; 1.06e-7 / 2.2e-9 ≈ 48×). The accumulated 144-step drift explains the observed GL sample-position shift (~1.4e-5) and why GL crosses the TF knot while proxy-Metal (~3e-7) stays safely on the pre-flip side. The fullscreen path (5.2) is the most sensitive of the three, as Update 17 already showed.

The amplitude difference is a property of each backend's proxy-mesh rasterization/interpolation arithmetic (attribute layout, densify pattern, thin triangles near the near-plane clip, per-vertex origin/step vs. analytic `localPos - cameraPos` ray reconstruction). The *cause* of both shifts is the same camera perturbation.

---

## 7. Correction to Update 18

| Update 18 claim | Update 19 finding |
|---|---|
| "Camera / near-plane / sample-distance / matrices constant across all 6 frames" | Camera position, near-plane, sample distance constant — but **view angle** changes 30 → 30.0000008 at frame 3→4 (hence the projection matrix changes); "constant" was a 6-sig-fig print artifact |
| "Metal proxy path is perfectly deterministic" | Not deterministic: ray geometry shifts at the frame 3→4 boundary (9-decimal `STEP` log), and even at 6 decimals a few samples shift 1 ulp in `t` in frames 4-6 |
| "The wheel event is not the cause" | Still correct |
| "The residual is GL's own frame-to-frame drift" | Not drift, not nondeterminism: a deterministic, harness-injected camera perturbation that GL amplifies ~50× more than proxy-Metal |
| "Metal on the proxy path is the parity reference" | Correct in *value* (i=144 matches GL's pre-flip frames) but not because proxy-Metal is immune — it merely shifts less |

---

## 8. Interpretation

- The i=144 residual is **not a Metal-vs-GL rendering difference**. Both backends produce `raw≈0.015427` at i=144 for the unperturbed camera (frames 1-3), and both change their ray geometry when the harness perturbs the camera (frames 4-6). GL's change happens to cross the scalar-1150 opacity knot; proxy-Metal's does not.
- The perturbation source is the VTK test harness itself: `vtkTesting::RegressionTest` re-renders the scene through `vtkWindowToImageFilter`, and that filter's float32 view-angle snapshot round-trips 30° to 30.0000008°. This affects **every** VTK test that compares against a baseline via W2IF; it is not specific to this test, the Metal backend, or this mapper.
- The comparison tool's last-frame-selection behavior (Update 18 section 6.3) is still what surfaces the residual, but the underlying cause is now fully explained and is reproducible in both backends at the same frame.

---

## 9. Reproduction

### 9.1 Source change (this update)

`Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`: `METAL_CAM` now prints with `std::setprecision(9)` (+ `#include <iomanip>`). Rebuild:

```sh
cmake --build build_macos_metal --target vtkRenderingVolumeCxxTests -j8
```

### 9.2 Capture

```sh
mkdir -p /tmp/bc/update20
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('/tmp/bc/dummy_baseline.png')"

VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0 \
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/update20/metal_proxy.log
```

(GL and fullscreen contrast logs unchanged from Updates 18/17.)

### 9.3 Sanity checks

```sh
# 3 pre-W2IF prints at exactly 30, 3 post-W2IF at 30.0000008 (9 sig figs):
rg 'METAL_CAM' /tmp/bc/update20/metal_proxy.log | sed 's/.*viewAngle=/viewAngle=/'

# STEP per-frame (frames 1-3 pass A identical; frames 4-6 pass A identical; boundary):
rg 'STEP px=\(372, 131\)' /tmp/bc/update20/metal_proxy.log | sed 's/.*localPos=/localPos=/'

# i=144 constant raw in Metal proxy:
rg 'SAMPLE px=\(372, 131\) i=144 ' /tmp/bc/update20/metal_proxy.log | sed 's/.*raw=/raw=/'

# GL i=144 flips at frame 3->4:
rg 'GL_SAMPLE px=\(372, 380\) i=144 ' /tmp/bc/update19/gl.log
```

### 9.4 Frame-identity check (within-stream only, due to CPU/shader stream interleaving)

```sh
python3 - <<'EOF'
import re
groups = []
cur = None
for l in open('/tmp/bc/update20/metal_proxy.log'):
    if 'SAMPLE px=(372, 131) i=0 ' in l:
        if cur: groups.append(cur)
        cur = [l]
    elif 'SAMPLE px=(372, 131) ' in l and cur is not None:
        cur.append(l)
if cur: groups.append(cur)
def body(line):
    m = re.search(r'DEBUG\s+(SAMPLE.*)', line)
    return m.group(1).rstrip('\n') if m else line
b = [[body(l) for l in g] for g in groups]
print('groups', len(groups))
for gi in range(12):
    d = sum(1 for x, y in zip(b[0], b[gi]) if x != y)
    print('g0 vs g%2d differing samples: %d' % (gi, d))
EOF
```

Expected: groups 0-5 (frames 1-3) differ only in timestamp; groups 6-11 (frames 4-6) differ from group 0 at a small number of samples (1-ulp `t` shifts), matching the frame 3→4 boundary seen in `STEP`.

---

## 10. Status and next steps

- Update 16: residual = one flipped TF sample i=144, caused by linear sample drift ~0.03 texel.
- Update 17: drift = ray-direction tilt from a float32 near-plane ray-origin difference between Metal's fullscreen reconstruction and GL's proxy interpolation; Step 1 = route camera-inside through the proxy path.
- Update 18: proxy path works and matches GL's pre-flip frames; concluded the residual was "GL's own drift" with a constant camera. **Refuted here.**
- **Update 19 (this):** the camera is not constant — the harness's `vtkWindowToImageFilter` (via `vtkTesting::RegressionTest`) perturbs the view angle 30 → 30.0000008° (float32 round-trip, verified to the printed digit) at the frame 3→4 boundary in both backends; all three paths change geometry at that boundary; the i=144 residual is GL's ~50× larger interpolation sensitivity to that same perturbation. The wheel event is irrelevant.

Remaining options:

1. **Accept and document.** The residual is a harness artifact, not a backend difference. Both backends agree at i=144 (`raw≈0.015427`) for the unperturbed camera; the W2IF comparison render is the only frame where they diverge, and only GL crosses the TF knot there.
2. **Remove the harness perturbation** (e.g., run the test with `-NoRerender` — `rtW2if->ShouldRerenderOff()` in vtkTesting.cxx:404-407 — or compare frames 1-3) and confirm the residual collapses to ~0, proving the causal chain end-to-end.
3. **Investigate GL's larger interpolation sensitivity** (why GL's per-step delta is ~50× proxy-Metal's) as a GL-side robustness question — a genuinely separate issue from the Metal-vs-GL comparison.

The debug override (`VTK_METAL_FULLSCREEN_CAMERA_INSIDE`), the 9-sig-fig `METAL_CAM` print, the dump hooks, and this report are the deliverables of this update.
