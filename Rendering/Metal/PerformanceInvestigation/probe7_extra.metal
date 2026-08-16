
// Appended by probe7b: minimal march for cost decomposition.
// March with NEAREST (single tap) instead of linear, otherwise identical.
fragment VolumeFragmentOut fragment_march_only_nearest(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;
  int maxSteps = max(1, int(ceil((tEnd - tStart) / stepSize)));

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < maxSteps; i++) {
    if (currentT >= tEnd - 1e-6) break;
    float s = volumeTexture.sample(sNearest, evalPoint, level(0)).r;
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Same loop arithmetic, but NO texture fetch at all: isolates loop/ALU cost.
fragment VolumeFragmentOut fragment_march_no_fetch(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;
  int maxSteps = max(1, int(ceil((tEnd - tStart) / stepSize)));

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < maxSteps; i++) {
    if (currentT >= tEnd - 1e-6) break;
    acc = max(acc, evalPoint.x * 0.01 + evalPoint.y * 0.001 + evalPoint.z * 0.0001);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// March with linear filter, IMPLICIT LOD (no level(0)).
fragment VolumeFragmentOut fragment_march_linear_implicit(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;
  int maxSteps = max(1, int(ceil((tEnd - tStart) / stepSize)));

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < maxSteps; i++) {
    if (currentT >= tEnd - 1e-6) break;
    float s = volumeTexture.sample(sVolume, evalPoint).r;
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Linear march with a FIXED global iteration count (no per-fragment divergence):
// removes loop-break divergence. N = fixedIterCount for every fragment.
fragment VolumeFragmentOut fragment_march_linear_fixedN(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float s = volumeTexture.sample(sVolume, evalPoint).r;
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Linear march with a FIXED global iteration count and EXPLICIT 2-deep
// software pipelining: fetch i+1 is issued before sample i is consumed, so
// two 3D fetches are in flight per warp instead of one.
fragment VolumeFragmentOut fragment_march_linear_pipe2(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  float sCur = volumeTexture.sample(sVolume, evalPoint).r;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 nextEval = evalPoint + evalStep;
    float sNext = volumeTexture.sample(sVolume, nextEval).r;
    acc = max(acc, sCur);
    evalPoint = nextEval;
    sCur = sNext;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Linear march, FIXED count, 3-deep software pipeline.
fragment VolumeFragmentOut fragment_march_linear_pipe3(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  float3 e0 = evalPoint;
  float3 e1 = e0 + evalStep;
  float3 e2 = e1 + evalStep;
  float s2 = volumeTexture.sample(sVolume, e2).r;
  float s1 = volumeTexture.sample(sVolume, e1).r;
  float s0 = volumeTexture.sample(sVolume, e0).r;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 e3 = e2 + evalStep;
    float s3 = volumeTexture.sample(sVolume, e3).r;
    acc = max(acc, max(s0, max(s1, s2)));
    e0 = e1; e1 = e2; e2 = e3;
    s0 = s1; s1 = s2; s2 = s3;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Debug: output stepSize and chord to red/green for readback.
fragment VolumeFragmentOut fragment_debug_steps(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  float rawDirLen = length(rayDir);
  if (!parallel) {
    if (rawDirLen < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= rawDirLen;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  float tEnd = t.y;
  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  if (tStart >= tEnd) { output.color = float4(0.0, 0.0, 1.0, 1.0); return output; }
  int maxSteps = max(1, int(ceil((tEnd - tStart) / stepSize)));
  int n = 0;
  float currentT = tStart;
  for (int i = 0; i < maxSteps; i++) {
    if (currentT >= tEnd - 1e-6) break;
    n++;
    currentT += stepSize;
  }
  int r = n & 255;
  int g = (n >> 8) & 255;
  // red = n%256, green = n/256, blue = stepSize / 0.002
  output.color = float4(float(r) / 255.0, float(g) / 255.0, clamp(stepSize / 0.002, 0.0, 1.0), 1.0);
  return output;
}

// Counts iterations per pixel via atomic accumulation (ground truth).
fragment VolumeFragmentOut fragment_count_steps(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    device atomic_uint* counters [[buffer(6)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0,0.0,0.0,1.0); return output; }
  float tEnd = t.y;
  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  int maxSteps = max(1, int(ceil((tEnd - tStart) / stepSize)));
  int n = 0;
  float currentT = tStart;
  for (int i = 0; i < maxSteps; i++) {
    if (currentT >= tEnd - 1e-6) break;
    n++;
    currentT += stepSize;
  }
  atomic_fetch_add_explicit(&counters[0], (uint)n, memory_order_relaxed);
  atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&counters[2], (uint)(n > 0 ? 1 : 0), memory_order_relaxed);
  int r = n & 255;
  int g = (n >> 8) & 255;
  output.color = float4(float(r) / 255.0, float(g) / 255.0, 0.0, 1.0);
  return output;
}


// Nearest march with a FIXED global iteration count: isolates raw nearest-tap
// throughput with zero loop divergence.
fragment VolumeFragmentOut fragment_march_nearest_fixedN(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float s = volumeTexture.sample(sNearest, evalPoint).r;
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Manual trilinear via 8 nearest fetches, fixed count.
fragment VolumeFragmentOut fragment_march_manual_trilinear(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 tc = evalPoint * texelCount;
    float3 f = fract(tc);
    float3 g = 1.0 - f;
    float3 ntc = float3(1.0/texelCount.x, 1.0/texelCount.y, 1.0/texelCount.z);
    float3 c000 = evalPoint - f * ntc;
    float v = 0.0;
    v += g.x * g.y * g.z * volumeTexture.sample(sNearest, c000, level(0)).r;
    v += f.x * g.y * g.z * volumeTexture.sample(sNearest, c000 + float3(ntc.x, 0.0, 0.0), level(0)).r;
    v += g.x * f.y * g.z * volumeTexture.sample(sNearest, c000 + float3(0.0, ntc.y, 0.0), level(0)).r;
    v += g.x * g.y * f.z * volumeTexture.sample(sNearest, c000 + float3(0.0, 0.0, ntc.z), level(0)).r;
    v += f.x * f.y * g.z * volumeTexture.sample(sNearest, c000 + float3(ntc.x, ntc.y, 0.0), level(0)).r;
    v += f.x * g.y * f.z * volumeTexture.sample(sNearest, c000 + float3(ntc.x, 0.0, ntc.z), level(0)).r;
    v += g.x * f.y * f.z * volumeTexture.sample(sNearest, c000 + float3(0.0, ntc.y, ntc.z), level(0)).r;
    v += f.x * f.y * f.z * volumeTexture.sample(sNearest, c000 + float3(ntc.x, ntc.y, ntc.z), level(0)).r;
    acc = max(acc, v);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Linear march fixed-N with clamp_to_zero sampler (tests edge-clamp cost).
// sVolumeClampZero comes from the base MetalShaders.metal (constexpr sampler).
fragment VolumeFragmentOut fragment_march_linear_clampZero(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float s = volumeTexture.sample(sVolumeClampZero, evalPoint).r;
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Fixed-N march sampling a 2D array texture with manual slice-lerp trilinear.
fragment VolumeFragmentOut fragment_march_linear_2Darray(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture2d_array<float> volumeArray [[texture(15)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeArray.get_width(), volumeArray.get_height(), volumeArray.get_array_size());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float depth = (float)volumeArray.get_array_size();
  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float zf = evalPoint.z * depth;              // in slice units (0..depth)
    float zc = zf - 0.5;                         // center-aligned
    float zClamped = clamp(zc, 0.0, (float)(depth - 1));
    int z0 = (int)floor(zClamped);
    int z1 = min(z0 + 1, (int)(depth - 1));
    float fz = zClamped - (float)z0;
    float2 xy = float2(evalPoint.x, evalPoint.y);
    float v0 = volumeArray.sample(sVolume, xy, z0).r;
    float v1 = volumeArray.sample(sVolume, xy, z1).r;
    float s = mix(v0, v1, fz);
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Linear march fixed-N with repeat sampler (no clamp cost; edges wrong, just for timing).
constant sampler sVolumeRepeat = sampler(coord::normalized, address::repeat, filter::linear);
fragment VolumeFragmentOut fragment_march_linear_repeat(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float s = volumeTexture.sample(sVolumeRepeat, evalPoint).r;
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Linear march fixed-N with nearest-filtered 3D texture but bilinear manual XY only (z-nearest).
fragment VolumeFragmentOut fragment_march_xybilinear_znearest(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    // 2D bilinear (linear xy) + nearest z: sample at z-floor and lerp? No - just sample
    // with the 3D texture's linear filter on xy only is impossible; use nearest 3D at two
    // z-neighbor coords? Simplest timing test: 2x nearest samples.
    float3 c = clamp(evalPoint, float3(0.0), float3(1.0));
    float s = 0.5 * (volumeTexture.sample(sNearest, c).r + volumeTexture.sample(sNearest, c).r);
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Isolates raw texture-unit throughput: same coordinate every iteration (no
// address math in loop), fixed count, linear filter.
fragment VolumeFragmentOut fragment_fixedpoint_linear(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 localPos = in.localPos;
  float3 evalPoint = float3(0.5, 0.5, 0.5) + (fract(localPos) - 0.5) * 1e-4;
  float acc = 0.0;
  float drift = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 p = evalPoint + float3(drift);
    drift += 1e-6;
    float s = volumeTexture.sample(sVolume, p).r;
    acc = max(acc, s);
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Same fixed-point loop with nearest filter.
fragment VolumeFragmentOut fragment_fixedpoint_nearest(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 localPos = in.localPos;
  float3 evalPoint = float3(0.5, 0.5, 0.5) + (fract(localPos) - 0.5) * 1e-4;
  float acc = 0.0;
  float drift = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 p = evalPoint + float3(drift);
    drift += 1e-6;
    float s = volumeTexture.sample(sNearest, p).r;
    acc = max(acc, s);
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Linear march, FIXED uniform count, UNCONDITIONAL fetch, select-guarded
// accumulation (no data-dependent break -> no divergence), image-correct.
fragment VolumeFragmentOut fragment_march_linear_select(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float s = volumeTexture.sample(sVolume, evalPoint).r;
    bool inside = currentT < tEnd;
    acc = inside ? max(acc, s) : acc;
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// Fixed-N march advancing along +X only (synthetic): tests if per-sample cost
// depends on march direction vs the swizzled layout.
fragment VolumeFragmentOut fragment_march_xdir_linear(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 localPos = in.localPos;
  float3 start = float3(0.001, 0.5, 0.5) + (fract(localPos) - 0.5) * 1e-3;
  float3 evalPoint = start;
  float acc = 0.0;
  float dx = (1.0 - 0.002) / (float)fixedIterCount;
  for (int i = 0; i < fixedIterCount; i++) {
    float s = volumeTexture.sample(sVolume, evalPoint).r;
    acc = max(acc, s);
    evalPoint.x += dx;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// v18 + iteration counter to verify the loop really runs.
fragment VolumeFragmentOut fragment_march_xdir_linear_counted(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]],
    device atomic_uint* counters [[buffer(6)]]) {

  VolumeFragmentOut output;
  float3 localPos = in.localPos;
  float3 start = float3(0.001, 0.5, 0.5) + (fract(localPos) - 0.5) * 1e-3;
  float3 evalPoint = start;
  float acc = 0.0;
  float dx = (1.0 - 0.002) / (float)fixedIterCount;
  for (int i = 0; i < fixedIterCount; i++) {
    float s = volumeTexture.sample(sVolume, evalPoint).r;
    acc = max(acc, s);
    evalPoint.x += dx;
    atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// ============================================================================
// probe7b decomposition: mirror of the real marchVolumeUnified composite loop
// (DICOM bench config) with fc_decomp bit gates so each per-iteration cost can
// be stripped at compile time:
//   &1  : no 2D transfer-function fetch (colorOpacity = const 0.6)
//   &2  : no 3D volume fetch (rawScalar = 0.5, no prefetch)
//   &4  : no CTP-bounds exit test
//   &8  : no composite accumulation
//   &16 : no exit breaks (geometric loop, all maxSteps iterations)
// Default (0) should reproduce the app baseline (~98ms in the probe).
// ============================================================================
fragment VolumeFragmentOut fragment_march_real_decomp(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_decomp [[buffer(7)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  const int decomp = fc_decomp;
  const bool noTF   = (decomp & 1) != 0;
  const bool noVol  = (decomp & 2) != 0;
  const bool noCTP  = (decomp & 4) != 0;
  const bool noComp = (decomp & 8) != 0;
  const bool noExit = (decomp & 16) != 0;

  half scalarScale = half(1.0 / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4h));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float tStart = dot(s.entryPoint - cameraPos, rayDir);
  float tEnd = s.totalBoxT;
  float jitter = (volumeUniforms.useJittering > 0.5
      ? (volumeUniforms.useIGNJitter > 0.5
            ? sampleIGNJitter(in.position.xy, volumeUniforms.jitterBlockSize)
            : sampleJitterNoise(in.position.xy, volumeUniforms.viewportSize.y))
      : 1.0) * stepSize;
  float firstT = jitter;
  float3 stepVec = rayDir * stepSize;
  float3 currentPoint = rayOrigin + rayDir * (tStart + firstT);
  float currentT = firstT;

  int maxSteps = max(1, int(ceil((tEnd - firstT) / stepSize)));
  if (noExit) {
    maxSteps = max(maxSteps, fixedIterCount);
  }

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float prefetchScalar = noVol ? 0.5
      : sampleVolumeScalar(volumeTexture, evalPoint);
  bool prefetchValid = true;
  bool seenInBounds = false;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;

  for (int i = 0; i < maxSteps; i++) {
    if (!noCTP) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) {
          break;
        }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
        prefetchValid = false;
      } else {
        seenInBounds = true;
      }
    }

    bool needsFetch = !prefetchValid;
    float rawScalar = needsFetch
        ? (noVol ? 0.5 : sampleVolumeScalar(volumeTexture, evalPoint))
        : prefetchScalar;

    half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias);

    half4 colorOpacity = noTF
        ? half4(0.5h, 0.5h, 0.5h, 0.6h)
        : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    half sampleOpacity = colorOpacity.a;

    if (!noComp) {
      half3 sampleColor = colorOpacity.rgb;
      half weight = 1.0h - accumulatedOpacity;
      accumulatedColor += weight * (sampleColor * sampleOpacity);
      accumulatedOpacity += weight * sampleOpacity;
    }

    currentPoint += stepVec;
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;

    if (i + 1 < maxSteps) {
      prefetchScalar = noVol ? 0.5 : sampleVolumeScalar(volumeTexture, evalPoint);
      prefetchValid = true;
    }

    if (!noExit) {
      if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) {
        break;
      }
      if (currentT >= s.tTerminateMax) {
        break;
      }
    }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// v21: clone of fragment_march_linear_fixedN (v7) but clamp the sample
// coordinate to [0,1] every iteration. Tests whether out-of-range coordinates
// (the sampler's clamp_to_edge slow path) are what cause the ~100ms floor.
fragment VolumeFragmentOut fragment_march_linear_clamp(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 c = clamp(evalPoint, float3(0.0), float3(1.0));
    float s = volumeTexture.sample(sVolume, c).r;
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// v22: clone of fragment_march_nearest_fixedN (v8) but clamp the sample
// coordinate to [0,1] every iteration. Same clamp test for the nearest path.
fragment VolumeFragmentOut fragment_march_nearest_clamp(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 c = clamp(evalPoint, float3(0.0), float3(1.0));
    float s = volumeTexture.sample(sNearest, c).r;
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// v23: byte-for-byte copy of v15's loop body (double-sample + clamp) under a
// different name, to test whether the 10x speedup is from the loop body or
// from something about function identity/source order.
fragment VolumeFragmentOut fragment_march_xybilinear_znearest_clone(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 c = clamp(evalPoint, float3(0.0), float3(1.0));
    float s = 0.5 * (volumeTexture.sample(sNearest, c).r + volumeTexture.sample(sNearest, c).r);
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// v24: v22 (clamped single nearest) but with the redundant double sample
// restored, i.e. v22 + 0.5*(A+A). Tests whether the double-sample expression
// is what triggers the fast path under runtime compilation.
fragment VolumeFragmentOut fragment_march_nearest_clamp_double(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  float2 t = intersectBox(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) { output.color = float4(0.0); return output; }
  float tEnd = t.y;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * currentT) * boundsSize, 1.0)).xyz;
  float3 evalPoint = texLocalPos * ctpScale + ctpOffset;

  float acc = 0.0;
  for (int i = 0; i < fixedIterCount; i++) {
    float3 c = clamp(evalPoint, float3(0.0), float3(1.0));
    float s = 0.5 * (volumeTexture.sample(sNearest, c).r + volumeTexture.sample(sNearest, c).r);
    acc = max(acc, s);
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
  }
  output.color = float4(float3(acc), 1.0);
  return output;
}

// ============================================================================
// v25: standalone clone of marchVolumeUnified's fc_marchVariant>=6 unrolled
// branch (batch-8 fetches + PROC consume with tEnd-latch, bounds clamp+refill,
// latch exits + suppressAccum), in the decomp fragment's self-contained
// function context. Isolates whether the unrolled consume MACHINERY itself is
// the ~30ms (75 vs 45.5) unrolled-vs-decomp gap, or whether it only costs that
// much inside fragment_volume_main's register-heavy context.
// ============================================================================
fragment VolumeFragmentOut fragment_march_decomp_unrolled(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v25mode [[buffer(8)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  const int mode = fc_v25mode;
  const bool usePrefetch = (mode & 1) != 0;    // prefetch-ahead-1 instead of batch-8
  const bool noTEndLatch = (mode & 2) != 0;    // drop the tEnd-latch at consume top
  const bool noBatchAbort = (mode & 4) != 0;   // decomp-style clamp+needsFetch instead of skip+refill
  const bool useBreak = (mode & 8) != 0;       // direct breaks instead of latch exits

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  half scalarScale = half(1.0 / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4h));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;
  float3 stepVec = rayDir * stepSize;

  float tStart = dot(s.entryPoint - cameraPos, rayDir);
  float tEnd = s.totalBoxT;
  float tTerminateMax = s.tTerminateMax;

  float3 currentPoint = rayOrigin + rayDir * (tStart + stepSize);
  float currentT = stepSize;
  int maxSteps = max(1, int(ceil((tEnd - stepSize) / stepSize)));

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;
  bool seenInBounds = false;
  bool marchOpaque = false;
  bool marchDone = false;
  float prefetchScalar = 0.0;
  bool prefetchValid = true;

  const int unrollN = 8;
  int i = 0;
  float bs[8];
  bs[0] = sampleVolumeScalar(volumeTexture, evalPoint);
  while (i < maxSteps)
  {
    const int nBatch = (usePrefetch ? 1 : ((maxSteps - i >= unrollN) ? unrollN : (maxSteps - i)));
    if (!usePrefetch) {
      if (nBatch > 0) { bs[0] = sampleVolumeScalar(volumeTexture, evalPoint); }
      if (nBatch > 1) { bs[1] = sampleVolumeScalar(volumeTexture, evalPoint + evalStep); }
      if (nBatch > 2) { bs[2] = sampleVolumeScalar(volumeTexture, evalPoint + 2.0 * evalStep); }
      if (nBatch > 3) { bs[3] = sampleVolumeScalar(volumeTexture, evalPoint + 3.0 * evalStep); }
      if (nBatch > 4) { bs[4] = sampleVolumeScalar(volumeTexture, evalPoint + 4.0 * evalStep); }
      if (nBatch > 5) { bs[5] = sampleVolumeScalar(volumeTexture, evalPoint + 5.0 * evalStep); }
      if (nBatch > 6) { bs[6] = sampleVolumeScalar(volumeTexture, evalPoint + 6.0 * evalStep); }
      if (nBatch > 7) { bs[7] = sampleVolumeScalar(volumeTexture, evalPoint + 7.0 * evalStep); }
    }
    bool refillNeeded = false;
#define UPROC(_j) \
    { \
      prefetchScalar = bs[_j]; \
      prefetchValid = true; \
      bool batchAbort = false; \
      if (!noTEndLatch) { \
        if (currentT >= tEnd - 1e-6) { marchDone = true; } \
      } \
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) || \
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) { \
        if (seenInBounds) { \
          if (useBreak) { break; } \
          marchDone = true; \
        } \
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0)); \
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset); \
        prefetchValid = false; \
        if (!noBatchAbort) { \
          currentPoint += stepVec; \
          currentT += stepSize; \
          texLocalPos += texStep; \
          evalPoint += evalStep; \
          batchAbort = true; \
        } \
      } else { \
        seenInBounds = true; \
      } \
      if (batchAbort) { \
        refillNeeded = true; \
      } else { \
        bool needsFetch = !prefetchValid; \
        float rawScalar = needsFetch ? sampleVolumeScalar(volumeTexture, evalPoint) : prefetchScalar; \
        half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias); \
        half4 colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5)); \
        half sampleOpacity = colorOpacity.a; \
        half3 sampleColor = colorOpacity.rgb; \
        half weight = 1.0h - accumulatedOpacity; \
        const bool suppressAccum = useBreak ? false : (marchOpaque || marchDone); \
        accumulatedColor += suppressAccum ? 0.0h : weight * (sampleColor * sampleOpacity); \
        accumulatedOpacity += suppressAccum ? 0.0h : weight * sampleOpacity; \
        currentPoint += stepVec; \
        currentT += stepSize; \
        texLocalPos += texStep; \
        evalPoint += evalStep; \
        if (useBreak) { \
          if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; } \
          if (currentT >= tTerminateMax) { break; } \
        } else { \
          if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { marchOpaque = true; } \
          if (currentT >= tTerminateMax) { marchDone = true; } \
        } \
      } \
      i++; \
    }
    if (nBatch > 0) { UPROC(0) }
    if (nBatch > 1 && !refillNeeded) { UPROC(1) }
    if (nBatch > 2 && !refillNeeded) { UPROC(2) }
    if (nBatch > 3 && !refillNeeded) { UPROC(3) }
    if (nBatch > 4 && !refillNeeded) { UPROC(4) }
    if (nBatch > 5 && !refillNeeded) { UPROC(5) }
    if (nBatch > 6 && !refillNeeded) { UPROC(6) }
    if (nBatch > 7 && !refillNeeded) { UPROC(7) }
#undef UPROC
    if (usePrefetch) {
      if (i < maxSteps) { bs[0] = sampleVolumeScalar(volumeTexture, evalPoint); }
      if (refillNeeded) { bs[0] = sampleVolumeScalar(volumeTexture, evalPoint); }
    }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// ============================================================================
// v27: verbatim copy of fragment_march_real_decomp (prefetch-ahead-1) with an
// optional `float bs[8]` array threaded through the prefetch. Tests whether the
// bs[8] array allocation (local memory / register pressure) is what costs the
// ~30ms in v25 batch mode. mode&1 = array plumbing (bs[0] store/load),
// mode&2 = variable index bs[i % 8] to force true array semantics.
// ============================================================================
fragment VolumeFragmentOut fragment_march_decomp_array(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v25mode [[buffer(8)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  const int mode = fc_v25mode;
  const bool useArr = (mode & 1) != 0;
  const bool varIdx = (mode & 2) != 0;
  const bool pure = (mode & 4) != 0;

  half scalarScale = half(1.0 / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4h));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float tStart = dot(s.entryPoint - cameraPos, rayDir);
  float tEnd = s.totalBoxT;
  float jitter = (volumeUniforms.useJittering > 0.5
      ? (volumeUniforms.useIGNJitter > 0.5
            ? sampleIGNJitter(in.position.xy, volumeUniforms.jitterBlockSize)
            : sampleJitterNoise(in.position.xy, volumeUniforms.viewportSize.y))
      : 1.0) * stepSize;
  float firstT = jitter;
  float3 stepVec = rayDir * stepSize;
  float3 currentPoint = rayOrigin + rayDir * (tStart + firstT);
  float currentT = firstT;

  int maxSteps = max(1, int(ceil((tEnd - firstT) / stepSize)));

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float prefetchScalar = sampleVolumeScalar(volumeTexture, evalPoint);
  bool prefetchValid = true;
  bool seenInBounds = false;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;

  float bs[8];
  if (useArr) {
    bs[0] = prefetchScalar;
  }
  if (pure) {
    prefetchScalar = sampleVolumeScalar(volumeTexture, evalPoint);
    prefetchValid = true;
  }

  for (int i = 0; i < maxSteps; i++) {
    if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
        any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
      if (seenInBounds) {
        break;
      }
      texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      prefetchValid = false;
    } else {
      seenInBounds = true;
    }

    if (useArr) {
      prefetchScalar = varIdx ? bs[i % 8] : bs[0];
      prefetchValid = true;
    }
    bool needsFetch = !prefetchValid;
    float rawScalar = needsFetch
        ? sampleVolumeScalar(volumeTexture, evalPoint)
        : prefetchScalar;

    half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias);
    half4 colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    half sampleOpacity = colorOpacity.a;
    half3 sampleColor = colorOpacity.rgb;
    half weight = 1.0h - accumulatedOpacity;
    accumulatedColor += weight * (sampleColor * sampleOpacity);
    accumulatedOpacity += weight * sampleOpacity;

    currentPoint += stepVec;
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;

    if (pure) {
      if (i + 1 < maxSteps) {
        prefetchScalar = sampleVolumeScalar(volumeTexture, evalPoint);
        prefetchValid = true;
      }
    } else {
      if (i + 1 < maxSteps) {
        float v = sampleVolumeScalar(volumeTexture, evalPoint);
        if (useArr) {
          bs[varIdx ? (i % 8) : 0] = v;
        } else {
          prefetchScalar = v;
        }
        prefetchValid = true;
      }
    }

    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) {
      break;
    }
    if (currentT >= s.tTerminateMax) {
      break;
    }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// ============================================================================
// v28: verbatim copy of fragment_march_real_decomp with a single addition:
// `float bs[8];` declared at function scope and NEVER referenced. Decisive
// occupancy test -- does the mere stack allocation of a 32-byte array halve
// the fragment's throughput (94ms) vs the identical array-free loop (46ms)?
// ============================================================================
fragment VolumeFragmentOut fragment_march_decomp_deadarr(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v25mode [[buffer(8)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  const int mode = fc_v25mode;
  (void)mode;

  half scalarScale = half(1.0 / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4h));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float tStart = dot(s.entryPoint - cameraPos, rayDir);
  float tEnd = s.totalBoxT;
  float jitter = (volumeUniforms.useJittering > 0.5
      ? (volumeUniforms.useIGNJitter > 0.5
            ? sampleIGNJitter(in.position.xy, volumeUniforms.jitterBlockSize)
            : sampleJitterNoise(in.position.xy, volumeUniforms.viewportSize.y))
      : 1.0) * stepSize;
  float firstT = jitter;
  float3 stepVec = rayDir * stepSize;
  float3 currentPoint = rayOrigin + rayDir * (tStart + firstT);
  float currentT = firstT;

  int maxSteps = max(1, int(ceil((tEnd - firstT) / stepSize)));

  float bs[8];

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float prefetchScalar = sampleVolumeScalar(volumeTexture, evalPoint);
  bool prefetchValid = true;
  bool seenInBounds = false;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;

  for (int i = 0; i < maxSteps; i++) {
    if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
        any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
      if (seenInBounds) {
        break;
      }
      texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      prefetchValid = false;
    } else {
      seenInBounds = true;
    }

    bool needsFetch = !prefetchValid;
    float rawScalar = needsFetch
        ? sampleVolumeScalar(volumeTexture, evalPoint)
        : prefetchScalar;

    half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias);
    half4 colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    half sampleOpacity = colorOpacity.a;
    half3 sampleColor = colorOpacity.rgb;
    half weight = 1.0h - accumulatedOpacity;
    accumulatedColor += weight * (sampleColor * sampleOpacity);
    accumulatedOpacity += weight * sampleOpacity;

    currentPoint += stepVec;
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;

    if (i + 1 < maxSteps) {
      prefetchScalar = sampleVolumeScalar(volumeTexture, evalPoint);
      prefetchValid = true;
    }

    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) {
      break;
    }
    if (currentT >= s.tTerminateMax) {
      break;
    }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// ============================================================================
// v29: v27 but WITHOUT the fc_v25mode buffer(8) read. Isolates whether the
// v27 slowdown came from the dead `float bs[8]` array or from the extra
// constant buffer read.
// ============================================================================
fragment VolumeFragmentOut fragment_march_decomp_deadarr2(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 localPos = in.localPos;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel) {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);

  half scalarScale = half(1.0 / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4h));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float tStart = dot(s.entryPoint - cameraPos, rayDir);
  float tEnd = s.totalBoxT;
  float jitter = (volumeUniforms.useJittering > 0.5
      ? (volumeUniforms.useIGNJitter > 0.5
            ? sampleIGNJitter(in.position.xy, volumeUniforms.jitterBlockSize)
            : sampleJitterNoise(in.position.xy, volumeUniforms.viewportSize.y))
      : 1.0) * stepSize;
  float firstT = jitter;
  float3 stepVec = rayDir * stepSize;
  float3 currentPoint = rayOrigin + rayDir * (tStart + firstT);
  float currentT = firstT;

  int maxSteps = max(1, int(ceil((tEnd - firstT) / stepSize)));

  float bs[8];

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float prefetchScalar = sampleVolumeScalar(volumeTexture, evalPoint);
  bool prefetchValid = true;
  bool seenInBounds = false;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;

  for (int i = 0; i < maxSteps; i++) {
    if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
        any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
      if (seenInBounds) {
        break;
      }
      texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      prefetchValid = false;
    } else {
      seenInBounds = true;
    }

    bool needsFetch = !prefetchValid;
    float rawScalar = needsFetch
        ? sampleVolumeScalar(volumeTexture, evalPoint)
        : prefetchScalar;

    half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias);
    half4 colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    half sampleOpacity = colorOpacity.a;
    half3 sampleColor = colorOpacity.rgb;
    half weight = 1.0h - accumulatedOpacity;
    accumulatedColor += weight * (sampleColor * sampleOpacity);
    accumulatedOpacity += weight * sampleOpacity;

    currentPoint += stepVec;
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;

    if (i + 1 < maxSteps) {
      prefetchScalar = sampleVolumeScalar(volumeTexture, evalPoint);
      prefetchValid = true;
    }

    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) {
      break;
    }
    if (currentT >= s.tTerminateMax) {
      break;
    }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}
