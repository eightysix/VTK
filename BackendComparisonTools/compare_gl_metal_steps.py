#!/usr/bin/env python3
"""Compare Metal volume raycaster STEP composition against OpenGL raycast mapper.

Reads the two debug logs produced by:
  * Metal vertex/fragment: vtkMetalVolumeMapper DEBUG STEP px=...
  * GL mapper:            vtkOpenGLGPUVolumeRayCastMapper DEBUG GL_RAY px=... / GL_UNIFORMS

and replays both float32 chains to localize the source of the step-size
difference between the two backends.

Requires numpy. Tested against the headless paired-run logs.
"""

import re
import sys

import numpy as np

F = np.float32


def f32(v):
    return np.asarray(v, dtype=np.float32)


def glsl_length(v):
    v = f32(v)
    return F(np.sqrt(F(np.sum(F(v * v)))))


def glsl_normalize(v):
    v = f32(v)
    return F(v * F(1.0 / F(glsl_length(v))))


def parse_vec(tok, n):
    vals = re.findall(r"[-+0-9.eE]+", tok)
    return [F(float(x)) for x in vals[:n]]


def parse_mat(tok):
    vals = re.findall(r"[-+0-9.eE]+", tok)
    return np.array([F(float(x)) for x in vals[:16]], dtype=np.float32).reshape(4, 4)


def parse_metal(metal_path, mpix):
    rows = []
    for line in open(metal_path, errors="replace"):
        m = re.search(r"DEBUG STEP px=\((\d+), (\d+)\) (.*)$", line)
        if not m:
            continue
        if (int(m.group(1)), int(m.group(2))) != mpix:
            continue
        body = m.group(3)
        d = {}
        for k, paren, scalar in re.findall(r"(\w+)=(?:\(([^)]*)\)|([-+0-9.eE]+))", body):
            d[k] = paren if paren else scalar
        rows.append(
            dict(
                cameraVol=parse_vec(d["cameraVol"], 3),
                localPos=parse_vec(d["localPos"], 3),
                rayDir=parse_vec(d["rayDir"], 3),
                dirObj=parse_vec(d["dirObj"], 3),
                evalStep=parse_vec(d["evalStep"], 3),
                boundsSize=parse_vec(d["boundsSize"], 3),
                sampleDistanceWorld=parse_vec(d["sampleDistanceWorld"], 1)[0],
                ctpScale=parse_vec(d["ctpScale"], 3),
                ctpOffset=parse_vec(d["ctpOffset"], 3),
            )
        )
    return rows


def parse_gl(gl_path, gpix):
    grows = []
    gunif_lines = []
    for line in open(gl_path, errors="replace"):
        m = re.search(r"DEBUG GL_RAY px=\((\d+), (\d+)\) (.*)$", line)
        if m and (int(m.group(1)), int(m.group(2))) == gpix:
            body = m.group(3)
            d = dict((k, v) for k, v in re.findall(r"(\w+)=\(([^)]*)\)", body))
            grows.append(
                dict(
                    cam=parse_vec(d["cam"], 3),
                    origin=parse_vec(d["origin"], 3),
                    step=parse_vec(d["step"], 3),
                    vpos=parse_vec(d["vpos"], 3),
                    tex=parse_vec(d["tex"], 3),
                )
            )
        if (
            "DEBUG GL_UNIFORMS" in line
            or "invTexDataset=" in line
            or "cellToPoint=" in line
            or "eyePosObjs=" in line
        ):
            gunif_lines.append(line)

    g = None
    if gunif_lines:
        sm = re.search(r"sampleDist=([-+0-9.eE]+)", "".join(gunif_lines))
        invTex = None
        cellToPoint = None
        eyePos = None
        for l in gunif_lines:
            if "invTexDataset=" in l:
                invTex = parse_mat(l.split("invTexDataset=", 1)[1])
            if "cellToPoint=" in l:
                cellToPoint = parse_mat(l.split("cellToPoint=", 1)[1])
            if "eyePosObjs=" in l:
                em = re.search(r"eyePosObjs=\(([^)]*)\)", l)
                eyePos = parse_vec(em.group(1), 3)
        g = dict(
            sampleDist=F(float(sm.group(1))) if sm else None,
            invTexDataset=invTex,
            cellToPoint=cellToPoint,
            eyePosObjs=eyePos,
        )
    return grows, g


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("usage: compare_gl_metal_steps.py metal.log gl.log [--mpix x,y] [--gpix x,y]")
        sys.exit(1)
    metal_path = sys.argv[1]
    gl_path = sys.argv[2]
    mpix = (372, 131)
    gpix = (372, 380)
    for a in sys.argv[3:]:
        if a.startswith("--mpix"):
            mpix = tuple(int(x) for x in sys.argv[sys.argv.index(a) + 1].split(","))
        if a.startswith("--gpix"):
            gpix = tuple(int(x) for x in sys.argv[sys.argv.index(a) + 1].split(","))

    mrows = parse_metal(metal_path, mpix)
    grows, g = parse_gl(gl_path, gpix)
    if not mrows:
        print(f"no Metal STEP lines for pixel {mpix}")
        sys.exit(1)
    if not grows:
        print(f"no GL_RAY lines for pixel {gpix}")
        sys.exit(1)
    if g is None or g["invTexDataset"] is None or g["cellToPoint"] is None:
        print("no GL_UNIFORMS block found")
        sys.exit(1)

    invTex = np.asarray(g["invTexDataset"], dtype=np.float64)
    ctp = np.asarray(g["cellToPoint"], dtype=np.float64)
    adj = f32(invTex @ ctp)  # adjustedLin == invTexDataset * cellToPoint
    sd = g["sampleDist"]
    eye = f32(g["eyePosObjs"])
    n = min(len(mrows), len(grows))

    print(f"Metal pixel {mpix}, GL pixel {gpix}, {n} frames")
    print(f"GL sampleDist={sd:.10}  invTexDataset diag={np.diag(invTex)}")
    print(f"GL cellToPoint diag={np.diag(ctp)}  offset={ctp[:3, 3]}")
    print(f"GL eyePosObjs={eye}")
    print()

    print("1) logged per-frame values:")
    print(f"{'f':>2} {'rayDir_MT':>24} {'dirObj_MT':>24} {'evalStep_MT':>24} {'step_GL':>24}")
    for i in range(n):
        mr = mrows[i]
        gr = grows[i]
        print(f"{i+1:>2} "
              f"{'(' + ','.join('%+.9e' % float(x) for x in mr['rayDir']) + ')':>24} "
              f"{'(' + ','.join('%+.9e' % float(x) for x in mr['dirObj']) + ')':>24} "
              f"{'(' + ','.join('%+.9e' % float(x) for x in mr['evalStep']) + ')':>24} "
              f"{'(' + ','.join('%+.9e' % float(x) for x in gr['step']) + ')':>24}")
    print()

    print("2) verify dirObj == normalize(rayDir*boundsSize)  (Metal's own values):")
    for i in range(n):
        mr = mrows[i]
        S = f32(mr["boundsSize"])
        rp = glsl_normalize(f32(f32(mr["rayDir"]) * S))
        err = np.max(np.abs(f32(rp - mr["dirObj"])))
        print(f"   frame {i+1}: max|dirObj - normalize(rayDir*S)|={err:.3e}  (0 => exact)")
    print()

    print("3) verify Metal evalStep == (adjustedLin_GL * dirObj) * sampleDist")
    print("   (adjustedLin = invTexDataset*cellToPoint from GL; must equal Metal's adjustedLin):")
    for i in range(n):
        mr = mrows[i]
        step = f32(f32(adj @ (mr["dirObj"][0], mr["dirObj"][1], mr["dirObj"][2], 0.0))[:3] * mr["sampleDistanceWorld"])
        err = np.linalg.norm(f32(step - mr["evalStep"]))
        print(f"   frame {i+1}: |gl-adj*dirObj*sd - evalStep|={err:.3e}  (0 => matrices are float32-identical)")
    print()

    print("4) SWAP-IN: GL step replayed with METAL dirObj (i.e. GL matrices but Metal's direction):")
    for i in range(n):
        mr = mrows[i]
        gr = grows[i]
        step = f32(f32(adj @ (mr["dirObj"][0], mr["dirObj"][1], mr["dirObj"][2], 0.0))[:3] * sd)
        e_gl = np.linalg.norm(f32(step - gr["step"]))
        e_mt = np.linalg.norm(f32(step - mr["evalStep"]))
        print(f"   frame {i+1}: |swap - step_GL|={e_gl:.3e}   |swap - evalStep_MT|={e_mt:.3e}")
    print()

    print("5) GL chain replay: (adjustedLin * normalize(vpos-eye)) * sampleDist  vs  logged GL step:")
    print("   (23-bit vpos decode introduces ~1e-4 direction noise):")
    for i in range(n):
        gr = grows[i]
        rd = glsl_normalize(f32(gr["vpos"] - eye))
        step = f32(f32(adj @ (rd[0], rd[1], rd[2], 0.0))[:3] * sd)
        err = np.linalg.norm(f32(step - gr["step"]))
        print(f"   frame {i+1}: |replay - logged step|={err:.3e}  (noise-dominated)")
    print()

    print("6) two-step normalization identity normalize(normalize(x/S)*S) == normalize(x):")
    for i in range(n):
        gr = grows[i]
        S = f32(mrows[i]["boundsSize"])
        x = f32(gr["vpos"] - eye)
        direct = glsl_normalize(x)
        twostep = glsl_normalize(f32(glsl_normalize(f32(x / S)) * S))
        d = np.linalg.norm(f32(direct - twostep))
        print(f"   frame {i+1}: |direct - two-step|={d:.3e}  (==0 => volume-space normalize is NOT the tilt source)")
    print()

    print("7) does Metal p.rayDir equal normalize((vpos-eye)/S)? (==0 would mean same ray):")
    for i in range(n):
        gr = grows[i]
        S = f32(mrows[i]["boundsSize"])
        eye_S = f32(eye / S)
        vpos_S = f32(gr["vpos"] / S)
        expected = glsl_normalize(f32(vpos_S - eye_S))
        got = f32(mrows[i]["rayDir"])
        print(f"   frame {i+1}: |rayDir - normalize((vpos-eye)/S)|={np.linalg.norm(f32(got - expected)):.3e}")
    print()

    print("8) camera agreement: Metal cameraVol (9-digit) vs GL eyePosObjs/S (9-digit):")
    for i in range(n):
        S = f32(mrows[i]["boundsSize"])
        cam_S = f32(eye / S)
        dcam = np.linalg.norm(f32(f32(mrows[i]["cameraVol"]) - cam_S))
        print(f"   frame {i+1}: Metal cameraVol={mrows[i]['cameraVol']}  GL eyePosObjs/S={cam_S}  |diff|={dcam:.3e}")
    print()

    # This test renders through the FULLSCREEN pass (fragment_volume_fullscreen_main,
    # camera-inside): Metal localPos == s.entryPoint == cameraPos + rayDir*tStart
    # (near-plane ray origin), NOT the interpolated proxy vertex. GL's vpos/S is the
    # near-plane point from its densified proxy-mesh clip (in_vertexPos interpolated).
    # Both are "where the ray starts"; the small difference IS the residual cause.
    print("9) near-plane ray origin: Metal s.entryPoint=localPos (9-digit) vs GL vpos/S (23-bit, 7 digits):")
    for i in range(n):
        gr = grows[i]
        S = f32(mrows[i]["boundsSize"])
        vpos_S = f32(gr["vpos"] / S)
        dlp = np.linalg.norm(f32(f32(mrows[i]["localPos"]) - vpos_S))
        print(f"   frame {i+1}: Metal entryPt={mrows[i]['localPos']}  GL vpos/S={vpos_S}  |diff|={dlp:.3e}")
    print()

    # Metal is fullscreen here, so p.localPos == s.entryPoint (camera + rayDir*t).
    # Then normalize(localPos - camVol) != rayDir is EXPECTED: float32 rounds
    # camera + rayDir*t (|d|~0.003 against camera~0.506, ulp 6e-8), and the
    # subtraction camera+delta - camera is exact, so the delta inherits up to
    # ~ulp(camera)/|delta| ~ 1e-5 rad of tilt, concentrated in the small delta
    # components (x). It is a logging-space artifact, not the GL-MT difference.
    print("10) rayDir vs normalize(localPos - camVol)  (Metal fullscreen: localPos=s.entryPoint):")
    for i in range(n):
        rd_local = glsl_normalize(f32(f32(mrows[i]["localPos"]) - f32(mrows[i]["cameraVol"])))
        rd_vpos = glsl_normalize(f32(f32(mrows[i]["localPos"]) - f32(eye / f32(mrows[i]["boundsSize"]))))
        print(f"   frame {i+1}: Metal rayDir={mrows[i]['rayDir']}")
        print(f"            normalize(localPos - camVol)   = {rd_local}")
        print(f"            normalize(localPos - eyePosObj/S) = {rd_vpos}")
    print()

    print("SUMMARY:")
    if n == 0:
        return
    mr = mrows[0]
    drel = 100.0 * (mr["evalStep"][1] - grows[0]["step"][1]) / float(mr["evalStep"][1])
    ang = np.linalg.norm(f32(glsl_normalize(f32(grows[0]["step"])) - glsl_normalize(f32(mr["evalStep"]))))
    dcam = np.linalg.norm(f32(f32(mr["cameraVol"]) - f32(eye / f32(mr["boundsSize"]))))
    print(f"  - step vectors differ mostly in y (up to {drel:.2f}% of the y-component);")
    print(f"    ray direction angle between backends: {ang:.2e} rad ({np.degrees(ang)*3600:.1f} arcsec)")
    print(f"  - matrices (invTexDataset*cellToPoint), sampleDist, AND the camera are bit-identical")
    print(f"    (swap-in test ~0; cameraVol vs eyePosObjs/S |diff|={dcam:.1e})")
    print(f"  - the whole step difference is caused by the ray direction, which differs because the two")
    print(f"    backends' near-plane ray ORIGINS differ by ~2-5e-7 volume (=4-11e-5 object units,")
    print(f"    ~0.03 texel in 512), 7-17x above the GL 23-bit decode noise floor.")
    print(f"  - this test renders via Metal's FULLSCREEN pass (camera inside): Metal origin =")
    print(f"    s.entryPoint = cameraPos + reconstructRayDir(NDC)*t (float32); GL origin = interpolated")
    print(f"    near-plane vertexPos of its densified ClipConvexPolyData proxy. Equivalent setups that")
    print(f"    differ at float32 level -> tiny ray tilt -> step tilt -> sub-texel sample drift -> the")
    print(f"    single TF flip at i=144 (update 16).")


if __name__ == "__main__":
    main()
