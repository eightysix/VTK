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
  padding: f32,
}


@group(0) @binding(0) var<uniform> sceneTransform: SceneTransform;

@group(1) @binding(0) var<uniform> volumeUniforms: VolumeMapperUniforms;
@group(1) @binding(1) var volumeTexture: texture_3d<f32>;
@group(1) @binding(2) var transferFunctionTexture: texture_2d<f32>;
@group(1) @binding(3) var transferFunctionSampler: sampler;

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
  @location(0) localPos: vec3<f32>,
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) selectorId: vec4<u32>,
}

fn intersectBox(orig: vec3<f32>, dir: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
  let invDir = 1.0 / dir;
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

  let dims = vec3<i32>(textureDimensions(volumeTexture, 0));
  let cameraPos = volumeUniforms.cameraVolumePos.xyz;
  let stepSize = volumeUniforms.sampleDistance;

  // Compute analytical ray-box intersection to determine exact entry/exit points
  let startPoint = input.localPos;
  let rayDir = normalize(startPoint - cameraPos);
  let t = intersectBox(cameraPos, rayDir, vec3<f32>(0.0), vec3<f32>(1.0));

  if (t.x >= t.y) {
    output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return output;
  }

  let entryPoint = cameraPos + rayDir * t.x;
  let exitPoint = cameraPos + rayDir * t.y;
  let totalDist = length(exitPoint - entryPoint);
  let maxSteps = max(1, i32(ceil(totalDist / stepSize)));

  var currentPoint = entryPoint;
  var accumulatedColor = vec3<f32>(0.0);
  var accumulatedOpacity = 0.0;

  for (var i = 0; i < maxSteps; i = i + 1) {
    let texCoord = vec3<i32>(clamp(
      vec3<f32>(currentPoint) * vec3<f32>(dims),
      vec3<f32>(0.0),
      vec3<f32>(dims - vec3<i32>(1))
    ));
    let rawScalar = textureLoad(volumeTexture, texCoord, 0).r;

    let scalarNorm = clamp(
      (rawScalar - volumeUniforms.scalarMin) /
      (volumeUniforms.scalarMax - volumeUniforms.scalarMin),
      0.0, 1.0);

    let colorOpacity = textureSampleLevel(transferFunctionTexture, transferFunctionSampler, vec2<f32>(scalarNorm, 0.5), 0.0);

    let sampleOpacity = colorOpacity.a;
    let sampleColor = colorOpacity.rgb;

    accumulatedColor = accumulatedColor + (1.0 - accumulatedOpacity) * sampleColor * sampleOpacity;
    accumulatedOpacity = accumulatedOpacity + (1.0 - accumulatedOpacity) * sampleOpacity;

    if (accumulatedOpacity >= 0.95) {
      accumulatedOpacity = 1.0;
      break;
    }

    currentPoint = currentPoint + rayDir * stepSize;
  }

  output.color = vec4<f32>(accumulatedColor, accumulatedOpacity);
  return output;
}
