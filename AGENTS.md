# AGENTS.md

Currently working on metal-gl mismatch for volume raycast mapper targeting bit-identical image output for TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter between metal and gl (assuming gl as stable reference): refer to latest VolumeRayCastBackendComparisonFindingsUpdate in rendering/metal.

Continue iterating automatically until bit-identical output is achieved providing md status updates in rendering/metal at milestones.

## Build Commands

To build the project, run:

```
./ios_metal_build.sh --resume
```

For macOS (without tests), build with:

```
./macos_metal_build.sh --resume
```

To build with tests, add `--tests`:

```
./macos_metal_build.sh --resume --tests
```
