// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

struct SceneTransform {
  viewport: vec4<f32>,
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  normal: mat3x3<f32>,
  inverted_projection: mat4x4<f32>,
  flags: u32
}

struct VolumeMapperUniforms {
  worldToVolume: mat4x4<f32>,
  volumeToWorld: mat4x4<f32>,
  volumeBoundsMin: vec4<f32>,
  volumeBoundsMax: vec4<f32>,
  cameraVolumePos: vec4<f32>,
  sampleDistance: f32,
  scalarMin: f32,
  scalarMax: f32,
  useJittering: f32,
}

@group(0) @binding(0) var<uniform> sceneTransform: SceneTransform;

@group(1) @binding(0) var<uniform> volumeUniforms: VolumeMapperUniforms;
@group(1) @binding(1) var volumeTexture: texture_3d<f32>;
@group(1) @binding(2) var transferFunctionTexture: texture_2d<f32>;
@group(1) @binding(3) var transferFunctionSampler: sampler;
@group(1) @binding(4) var volumeSampler: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPos: vec3<f32>,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.position = sceneTransform.projection * sceneTransform.view * volumeUniforms.volumeToWorld * vec4<f32>(input.position, 1.0);

  let boundsSize = volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz;
  output.localPos = (input.position - volumeUniforms.volumeBoundsMin.xyz) / boundsSize;

  return output;
}

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPos: vec3<f32>,
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) selectorId: vec4<u32>,
}

const MAX_RAY_STEPS: i32 = 2000;

fn random(st: vec2<f32>) -> f32 {
  return fract(sin(dot(st.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn intersectBox(orig: vec3<f32>, dir: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
  let invDir = 1.0 / (dir + vec3<f32>(1e-8));
  let tbot = invDir * (boxMin - orig);
  let ttop = invDir * (boxMax - orig);
  let tmin = min(ttop, tbot);
  let tmax = max(ttop, tbot);
  let t0 = max(max(tmin.x, tmin.y), tmin.z);
  let t1 = min(min(tmax.x, tmax.y), tmax.z);
  return vec2<f32>(t0, t1);
}

@fragment
fn fragmentMain(input: FragmentInput) -> FragmentOutput {
  var output: FragmentOutput;
  output.selectorId = vec4<u32>(0u, 0u, 0u, 0u);

  let cameraPos = volumeUniforms.cameraVolumePos.xyz;
  let stepSize = volumeUniforms.sampleDistance;

  let startPoint = input.localPos;
  var rayDir = startPoint - cameraPos;
  let dirLength = length(rayDir);

  if (dirLength < 0.0001) {
    discard;
  }
  rayDir = rayDir / dirLength;

  let t = intersectBox(cameraPos, rayDir, vec3<f32>(0.0), vec3<f32>(1.0));

  let tStart = max(t.x, 0.0);
  if (tStart >= t.y) {
    discard;
  }

  let entryPoint = cameraPos + rayDir * tStart;
  let exitPoint = cameraPos + rayDir * t.y;
  let totalDist = length(exitPoint - entryPoint);
  let maxSteps = min(max(1, i32(ceil(totalDist / stepSize))), MAX_RAY_STEPS);

  var jitter = 0.0;
  if (volumeUniforms.useJittering > 0.5) {
      jitter = random(input.position.xy) * stepSize;
  }

  var currentPoint = entryPoint + (rayDir * jitter);
  var accumulatedColor = vec3<f32>(0.0);
  var accumulatedOpacity = 0.0;

  for (var i = 0; i < maxSteps; i = i + 1) {
    let texCoord = clamp(currentPoint, vec3<f32>(0.0), vec3<f32>(1.0));
    let rawScalar = textureSampleLevel(volumeTexture, volumeSampler, texCoord, 0.0).r;

    let scalarNorm = clamp(
      (rawScalar - volumeUniforms.scalarMin) /
      (volumeUniforms.scalarMax - volumeUniforms.scalarMin),
      0.0, 1.0);

    let colorOpacity = textureSampleLevel(transferFunctionTexture, transferFunctionSampler, vec2<f32>(scalarNorm, 0.5), 0.0);

    let sampleOpacity = colorOpacity.a;
    if (sampleOpacity > 0.001) {
      let sampleColor = colorOpacity.rgb;
      accumulatedColor = accumulatedColor + (1.0 - accumulatedOpacity) * sampleColor * sampleOpacity;
      accumulatedOpacity = accumulatedOpacity + (1.0 - accumulatedOpacity) * sampleOpacity;
    }

    if (accumulatedOpacity >= 0.95) {
      accumulatedOpacity = 1.0;
      break;
    }

    currentPoint = currentPoint + rayDir * stepSize;
  }

  output.color = vec4<f32>(accumulatedColor, accumulatedOpacity);
  return output;
}
