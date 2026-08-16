
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

// v32: phase-separated batch-8. Replicates the minimal_gap harness's winning
// pattern (metal_gap.m pipeline=8, 45.8 ms / 1.00 ns/sample vs GL 67.1 ms):
// issue ALL N volume fetches back-to-back (independent, all in flight), then
// issue ALL N transfer-function fetches back-to-back (independent of each
// other, each depends only on its own scalar), THEN run the serial composite
// chain. The production batch-8 (fc_marchVariant 6) already batches the
// volume fetches but interleaves each sample's dependent TF fetch + composite
// inside the consume; v32 tests whether phase-separating the TF fetch restores
// the harness's MLP benefit in the full DVR. fc_v32mode (buffer 8):
//   &1 = interleave TF fetch with consume (production-style control)
//   &2 = no TF fetch (isolate volume fetch + composite)
//   &4 = no composite (isolate volume + TF fetch)
//   &8 = no bounds clamp (bare march, minimal_gap style)
// ============================================================================
fragment VolumeFragmentOut fragment_march_phase_batch(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v32mode [[buffer(8)]]) {

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
  const int mode = fc_v32mode;
  const bool interleaveTF = (mode & 1) != 0;
  const bool noTF   = (mode & 2) != 0;
  const bool noComp = (mode & 4) != 0;
  const bool noClamp = (mode & 8) != 0;

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

  const int unrollN = 8;
  int i = 0;
  while (i < maxSteps)
  {
    const int nBatch = (maxSteps - i >= unrollN) ? unrollN : (maxSteps - i);
    float bs[8];
    if (nBatch > 0) { bs[0] = sampleVolumeScalar(volumeTexture, evalPoint); }
    if (nBatch > 1) { bs[1] = sampleVolumeScalar(volumeTexture, evalPoint + evalStep); }
    if (nBatch > 2) { bs[2] = sampleVolumeScalar(volumeTexture, evalPoint + 2.0 * evalStep); }
    if (nBatch > 3) { bs[3] = sampleVolumeScalar(volumeTexture, evalPoint + 3.0 * evalStep); }
    if (nBatch > 4) { bs[4] = sampleVolumeScalar(volumeTexture, evalPoint + 4.0 * evalStep); }
    if (nBatch > 5) { bs[5] = sampleVolumeScalar(volumeTexture, evalPoint + 5.0 * evalStep); }
    if (nBatch > 6) { bs[6] = sampleVolumeScalar(volumeTexture, evalPoint + 6.0 * evalStep); }
    if (nBatch > 7) { bs[7] = sampleVolumeScalar(volumeTexture, evalPoint + 7.0 * evalStep); }

    half4 co[8];
    if (nBatch > 0) {
      half scalarNorm = saturate(half(bs[0]) * scalarScale + scalarBias);
      co[0] = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    }
    if (nBatch > 1) {
      half scalarNorm = saturate(half(bs[1]) * scalarScale + scalarBias);
      co[1] = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    }
    if (nBatch > 2) {
      half scalarNorm = saturate(half(bs[2]) * scalarScale + scalarBias);
      co[2] = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    }
    if (nBatch > 3) {
      half scalarNorm = saturate(half(bs[3]) * scalarScale + scalarBias);
      co[3] = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    }
    if (nBatch > 4) {
      half scalarNorm = saturate(half(bs[4]) * scalarScale + scalarBias);
      co[4] = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    }
    if (nBatch > 5) {
      half scalarNorm = saturate(half(bs[5]) * scalarScale + scalarBias);
      co[5] = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    }
    if (nBatch > 6) {
      half scalarNorm = saturate(half(bs[6]) * scalarScale + scalarBias);
      co[6] = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    }
    if (nBatch > 7) {
      half scalarNorm = saturate(half(bs[7]) * scalarScale + scalarBias);
      co[7] = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
    }

    if (!noClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }

    bool batchDone = false;
    for (int j = 0; j < nBatch; j++) {
      half sampleOpacity = co[j].a;
      if (!noComp) {
        half3 sampleColor = co[j].rgb;
        half weight = 1.0h - accumulatedOpacity;
        accumulatedColor += weight * (sampleColor * sampleOpacity);
        accumulatedOpacity += weight * sampleOpacity;
      }
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { batchDone = true; break; }
      if (currentT >= tTerminateMax) { batchDone = true; break; }
    }
    i += nBatch;
    if (batchDone) { break; }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// v33: phase-separated batch-8, scalar-only (no arrays). Faithful port of
// minimal_gap's winning 8x unroll (metal_gap.m BuildUnrollBody, 45.8ms /
// 1.00 ns/sample vs GL 67.1ms): compute all positions, issue all 8 volume
// fetches back-to-back, then all 8 TF fetches back-to-back (each independent,
// all in flight), then the serial composite chain, ONE break check per batch.
// v32 (identical idea) used runtime-indexed co[]/bs[] loops which forced local
// memory (83ms); v33 uses constant-indexed scalar variables so the compiler
// keeps all fetches in registers like the harness. fc_v33mode (buffer 8):
//   &1 = interleave TF fetch with composite (production-style control)
//   &2 = skip TF fetch (constant color, isolate volume fetch + composite)
//   &4 = skip composite (isolate volume + TF fetch)
//   &8 = no bounds clamp (bare march, minimal_gap style)
// ============================================================================
fragment VolumeFragmentOut fragment_march_phase_batch_scalar(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v33mode [[buffer(8)]]) {

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
  const int mode = fc_v33mode;
  const bool interleaveTF = (mode & 1) != 0;
  const bool noTF   = (mode & 2) != 0;
  const bool noComp = (mode & 4) != 0;
  const bool noClamp = (mode & 8) != 0;

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

  const int unrollN = 8;
  int i = 0;
  bool batchDone = false;
  while (i < maxSteps)
  {
    const int nBatch = (maxSteps - i >= unrollN) ? unrollN : (maxSteps - i);
    float s0, s1, s2, s3, s4, s5, s6, s7;
    if (nBatch > 0) { s0 = sampleVolumeScalar(volumeTexture, evalPoint); }
    if (nBatch > 1) { s1 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep); }
    if (nBatch > 2) { s2 = sampleVolumeScalar(volumeTexture, evalPoint + 2.0 * evalStep); }
    if (nBatch > 3) { s3 = sampleVolumeScalar(volumeTexture, evalPoint + 3.0 * evalStep); }
    if (nBatch > 4) { s4 = sampleVolumeScalar(volumeTexture, evalPoint + 4.0 * evalStep); }
    if (nBatch > 5) { s5 = sampleVolumeScalar(volumeTexture, evalPoint + 5.0 * evalStep); }
    if (nBatch > 6) { s6 = sampleVolumeScalar(volumeTexture, evalPoint + 6.0 * evalStep); }
    if (nBatch > 7) { s7 = sampleVolumeScalar(volumeTexture, evalPoint + 7.0 * evalStep); }

    half4 c0, c1, c2, c3, c4, c5, c6, c7;
#define FETCHTF(_k) \
    { half scalarNorm = saturate(half(s##_k) * scalarScale + scalarBias); \
      c##_k = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h) \
                   : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5)); }
    if (nBatch > 0) { FETCHTF(0) }
    if (nBatch > 1) { FETCHTF(1) }
    if (nBatch > 2) { FETCHTF(2) }
    if (nBatch > 3) { FETCHTF(3) }
    if (nBatch > 4) { FETCHTF(4) }
    if (nBatch > 5) { FETCHTF(5) }
    if (nBatch > 6) { FETCHTF(6) }
    if (nBatch > 7) { FETCHTF(7) }
#undef FETCHTF

    if (!noClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }

#define COMPOSITE(_k) \
    { half sampleOpacity = c##_k.a; \
      if (!noComp) { \
        half3 sampleColor = c##_k.rgb; \
        half weight = 1.0h - accumulatedOpacity; \
        accumulatedColor += weight * (sampleColor * sampleOpacity); \
        accumulatedOpacity += weight * sampleOpacity; \
      } \
      currentPoint += stepVec; \
      currentT += stepSize; \
      texLocalPos += texStep; \
      evalPoint += evalStep; \
      if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { batchDone = true; } \
      if (currentT >= tTerminateMax) { batchDone = true; } }
#define COMPOSITE_I(_k) \
    { half4 ctf = interleaveTF ? c##_k : half4(0.0h); \
      half scalarNorm = saturate(half(s##_k) * scalarScale + scalarBias); \
      ctf = interleaveTF ? ctf : sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5)); \
      half sampleOpacity = ctf.a; \
      if (!noComp) { \
        half3 sampleColor = ctf.rgb; \
        half weight = 1.0h - accumulatedOpacity; \
        accumulatedColor += weight * (sampleColor * sampleOpacity); \
        accumulatedOpacity += weight * sampleOpacity; \
      } \
      currentPoint += stepVec; \
      currentT += stepSize; \
      texLocalPos += texStep; \
      evalPoint += evalStep; \
      if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { batchDone = true; } \
      if (currentT >= tTerminateMax) { batchDone = true; } }
    if (interleaveTF) {
      if (nBatch > 0) { COMPOSITE_I(0) }
      if (nBatch > 1 && !batchDone) { COMPOSITE_I(1) }
      if (nBatch > 2 && !batchDone) { COMPOSITE_I(2) }
      if (nBatch > 3 && !batchDone) { COMPOSITE_I(3) }
      if (nBatch > 4 && !batchDone) { COMPOSITE_I(4) }
      if (nBatch > 5 && !batchDone) { COMPOSITE_I(5) }
      if (nBatch > 6 && !batchDone) { COMPOSITE_I(6) }
      if (nBatch > 7 && !batchDone) { COMPOSITE_I(7) }
    } else {
      if (nBatch > 0) { COMPOSITE(0) }
      if (nBatch > 1 && !batchDone) { COMPOSITE(1) }
      if (nBatch > 2 && !batchDone) { COMPOSITE(2) }
      if (nBatch > 3 && !batchDone) { COMPOSITE(3) }
      if (nBatch > 4 && !batchDone) { COMPOSITE(4) }
      if (nBatch > 5 && !batchDone) { COMPOSITE(5) }
      if (nBatch > 6 && !batchDone) { COMPOSITE(6) }
      if (nBatch > 7 && !batchDone) { COMPOSITE(7) }
    }
#undef COMPOSITE
#undef COMPOSITE_I
    i += nBatch;
    if (batchDone) { break; }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// v34: exact minimal_gap scheduling (BuildUnrollBody) applied to the full DVR.
// v33 proved scalar-only phase-separated batches beat the production array
// consume (69 vs 75ms). v34 now replicates the harness's loop shape verbatim:
//   - batch condition `i + N <= maxSteps` (no per-batch nBatch tail inside)
//   - ONE break check per batch at top: `if (currentT >= tEnd - 1e-6f) break;`
//   - compute all 8 positions p0..p7 FIRST
//   - issue all 8 volume fetches back-to-back
//   - issue all 8 TF fetches back-to-back (independent of each other)
//   - serial composite chain (DVR is inherently serial, unlike the harness's
//     max-tree, but the fetches are all independent and in flight)
//   - ONE advance per batch: evalPoint += evalStep * 8 (loop-carried chain is
//     1-op, not 8 serial adds)
//   - scalar break-aware tail loop (harness style)
// Tests whether the 8-long serial advance chain in v33's consume (and the
// per-sample tTerminateMax/opacity checks) cost the remaining 24ms vs the
// harness's 45.8ms. fc_v34mode (buffer 8):
//   &1 = per-sample advance inside consume (v33-style control)
//   &2 = keep per-sample opacity/tTerminate breaks inside consume
//   &4 = no TF fetch (constant color)
//   &8 = no composite
//   &16 = no bounds clamp (bare march, sampler clamp_to_edge handles exits)
// ============================================================================
fragment VolumeFragmentOut fragment_march_phase_batch_sched(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v34mode [[buffer(8)]]) {

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
  const int mode = fc_v34mode;
  const bool perSampleAdvance = (mode & 1) != 0;
  const bool perSampleBreaks = (mode & 2) != 0;
  const bool noTF   = (mode & 4) != 0;
  const bool noComp = (mode & 8) != 0;
  const bool noClamp = (mode & 16) != 0;

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

  const int unrollN = 8;
  int i = 0;
  const int steps = maxSteps;
  for (; i + unrollN <= steps; i += unrollN)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (!noClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }

    float3 p0 = evalPoint;
    float3 p1 = evalPoint + evalStep * 1.0f;
    float3 p2 = evalPoint + evalStep * 2.0f;
    float3 p3 = evalPoint + evalStep * 3.0f;
    float3 p4 = evalPoint + evalStep * 4.0f;
    float3 p5 = evalPoint + evalStep * 5.0f;
    float3 p6 = evalPoint + evalStep * 6.0f;
    float3 p7 = evalPoint + evalStep * 7.0f;

    float s0 = sampleVolumeScalar(volumeTexture, p0);
    float s1 = sampleVolumeScalar(volumeTexture, p1);
    float s2 = sampleVolumeScalar(volumeTexture, p2);
    float s3 = sampleVolumeScalar(volumeTexture, p3);
    float s4 = sampleVolumeScalar(volumeTexture, p4);
    float s5 = sampleVolumeScalar(volumeTexture, p5);
    float s6 = sampleVolumeScalar(volumeTexture, p6);
    float s7 = sampleVolumeScalar(volumeTexture, p7);

    half4 c0 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s0) * scalarScale + scalarBias)), 0.5));
    half4 c1 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s1) * scalarScale + scalarBias)), 0.5));
    half4 c2 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s2) * scalarScale + scalarBias)), 0.5));
    half4 c3 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s3) * scalarScale + scalarBias)), 0.5));
    half4 c4 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s4) * scalarScale + scalarBias)), 0.5));
    half4 c5 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s5) * scalarScale + scalarBias)), 0.5));
    half4 c6 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s6) * scalarScale + scalarBias)), 0.5));
    half4 c7 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s7) * scalarScale + scalarBias)), 0.5));

    if (!noComp) {
      half w0 = 1.0h - accumulatedOpacity;
      accumulatedColor += w0 * (c0.rgb * c0.a);
      accumulatedOpacity += w0 * c0.a;
      half w1 = 1.0h - accumulatedOpacity;
      accumulatedColor += w1 * (c1.rgb * c1.a);
      accumulatedOpacity += w1 * c1.a;
      half w2 = 1.0h - accumulatedOpacity;
      accumulatedColor += w2 * (c2.rgb * c2.a);
      accumulatedOpacity += w2 * c2.a;
      half w3 = 1.0h - accumulatedOpacity;
      accumulatedColor += w3 * (c3.rgb * c3.a);
      accumulatedOpacity += w3 * c3.a;
      half w4 = 1.0h - accumulatedOpacity;
      accumulatedColor += w4 * (c4.rgb * c4.a);
      accumulatedOpacity += w4 * c4.a;
      half w5 = 1.0h - accumulatedOpacity;
      accumulatedColor += w5 * (c5.rgb * c5.a);
      accumulatedOpacity += w5 * c5.a;
      half w6 = 1.0h - accumulatedOpacity;
      accumulatedColor += w6 * (c6.rgb * c6.a);
      accumulatedOpacity += w6 * c6.a;
      half w7 = 1.0h - accumulatedOpacity;
      accumulatedColor += w7 * (c7.rgb * c7.a);
      accumulatedOpacity += w7 * c7.a;
    }

    if (perSampleAdvance) {
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      currentPoint += stepVec;
      currentT += stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
    } else {
      currentPoint += stepVec * 8.0f;
      currentT += stepSize * 8.0f;
      texLocalPos += texStep * 8.0f;
      evalPoint += evalStep * 8.0f;
    }
    if (perSampleBreaks) {
      if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
      if (currentT >= tTerminateMax) { break; }
    }
  }
  for (; i < steps; i++)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (!noClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s = sampleVolumeScalar(volumeTexture, evalPoint);
    half4 c = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s) * scalarScale + scalarBias)), 0.5));
    if (!noComp) {
      half w = 1.0h - accumulatedOpacity;
      accumulatedColor += w * (c.rgb * c.a);
      accumulatedOpacity += w * c.a;
    }
    currentPoint += stepVec;
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// v35: v34's harness scheduling minus the per-batch machinery the compiler
// cannot DCE when the mode is a runtime buffer. Same batch-8 loop shape
// (positions first, all volume fetches back-to-back, all TF fetches
// back-to-back, serial composite, ONE advance per batch, scalar tail) but:
//   - the entry bounds clamp is hoisted ONCE before the loop; the batch loop
//     and tail loop carry NO clamp check and NO seenInBounds flag (the exit
//     side is handled by the existing currentT >= tEnd break; the ray is a
//     straight line, so after the entry clamp it stays in [0,1] until tEnd)
//   - no currentPoint / texLocalPos loop-carried chains (dead after the loop;
//     only evalPoint + currentT are advanced)
// Gives the Metal compiler a pure fetch->TF->composite loop with nothing else
// to serialize or spill. fc_v35mode (buffer 8):
//   &2 = keep v34-style per-batch clamp (A/B within one shader)
//   &4 = no TF fetch
//   &8 = no composite
//   &16 = no clamp at all (overrides &2)
// ============================================================================
fragment VolumeFragmentOut fragment_march_phase_batch_lean(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v35mode [[buffer(8)]]) {

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
  const int mode = fc_v35mode;
  const bool noTF   = (mode & 4) != 0;
  const bool noComp = (mode & 8) != 0;
  const bool noClamp = (mode & 16) != 0;
  const bool inLoopClamp = ((mode & 2) != 0) && !noClamp;

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
  float tTerminateMax = s.tTerminateMax;

  float currentT = stepSize;
  int maxSteps = max(1, int(ceil((tEnd - stepSize) / stepSize)));

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + (rayOrigin + rayDir * (tStart + stepSize)) * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;

  // Entry clamp, hoisted once. Matches v34's first-batch behavior (out-of-bounds
  // before any in-bounds sample: clamp texLocalPos to [0,1] and recompute).
  if (!noClamp) {
    if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
        any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
      texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
    }
  }

  const int unrollN = 8;
  int i = 0;
  const int steps = maxSteps;
  for (; i + unrollN <= steps; i += unrollN)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      }
    }
    float3 p0 = evalPoint;
    float3 p1 = evalPoint + evalStep * 1.0f;
    float3 p2 = evalPoint + evalStep * 2.0f;
    float3 p3 = evalPoint + evalStep * 3.0f;
    float3 p4 = evalPoint + evalStep * 4.0f;
    float3 p5 = evalPoint + evalStep * 5.0f;
    float3 p6 = evalPoint + evalStep * 6.0f;
    float3 p7 = evalPoint + evalStep * 7.0f;

    float s0 = sampleVolumeScalar(volumeTexture, p0);
    float s1 = sampleVolumeScalar(volumeTexture, p1);
    float s2 = sampleVolumeScalar(volumeTexture, p2);
    float s3 = sampleVolumeScalar(volumeTexture, p3);
    float s4 = sampleVolumeScalar(volumeTexture, p4);
    float s5 = sampleVolumeScalar(volumeTexture, p5);
    float s6 = sampleVolumeScalar(volumeTexture, p6);
    float s7 = sampleVolumeScalar(volumeTexture, p7);

    half4 c0 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s0) * scalarScale + scalarBias)), 0.5));
    half4 c1 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s1) * scalarScale + scalarBias)), 0.5));
    half4 c2 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s2) * scalarScale + scalarBias)), 0.5));
    half4 c3 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s3) * scalarScale + scalarBias)), 0.5));
    half4 c4 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s4) * scalarScale + scalarBias)), 0.5));
    half4 c5 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s5) * scalarScale + scalarBias)), 0.5));
    half4 c6 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s6) * scalarScale + scalarBias)), 0.5));
    half4 c7 = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                    : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s7) * scalarScale + scalarBias)), 0.5));

    if (!noComp) {
      half w0 = 1.0h - accumulatedOpacity;
      accumulatedColor += w0 * (c0.rgb * c0.a);
      accumulatedOpacity += w0 * c0.a;
      half w1 = 1.0h - accumulatedOpacity;
      accumulatedColor += w1 * (c1.rgb * c1.a);
      accumulatedOpacity += w1 * c1.a;
      half w2 = 1.0h - accumulatedOpacity;
      accumulatedColor += w2 * (c2.rgb * c2.a);
      accumulatedOpacity += w2 * c2.a;
      half w3 = 1.0h - accumulatedOpacity;
      accumulatedColor += w3 * (c3.rgb * c3.a);
      accumulatedOpacity += w3 * c3.a;
      half w4 = 1.0h - accumulatedOpacity;
      accumulatedColor += w4 * (c4.rgb * c4.a);
      accumulatedOpacity += w4 * c4.a;
      half w5 = 1.0h - accumulatedOpacity;
      accumulatedColor += w5 * (c5.rgb * c5.a);
      accumulatedOpacity += w5 * c5.a;
      half w6 = 1.0h - accumulatedOpacity;
      accumulatedColor += w6 * (c6.rgb * c6.a);
      accumulatedOpacity += w6 * c6.a;
      half w7 = 1.0h - accumulatedOpacity;
      accumulatedColor += w7 * (c7.rgb * c7.a);
      accumulatedOpacity += w7 * c7.a;
    }

    currentT += stepSize * 8.0f;
    evalPoint += evalStep * 8.0f;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }
  for (; i < steps; i++)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      }
    }
    float s = sampleVolumeScalar(volumeTexture, evalPoint);
    half4 c = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s) * scalarScale + scalarBias)), 0.5));
    if (!noComp) {
      half w = 1.0h - accumulatedOpacity;
      accumulatedColor += w * (c.rgb * c.a);
      accumulatedOpacity += w * c.a;
    }
    currentT += stepSize;
    evalPoint += evalStep;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// v36: same harness scheduling as v34 but a 16-wide batch with INLINE sample
// positions. The harness's 16x unroll regressed (45.8 -> 51.6ms) because it
// kept 16 float3 position variables live (48 registers) alongside 16 results;
// v36 passes the position expression directly to sampleVolumeScalar so each
// address dies at issue and only the 16 scalar results stay live - testing
// whether a wider batch can beat 8x in the full-DVR context (the app has only
// ~70k rays vs the harness's 160k, so per-warp ILP matters more here).
// fc_v36mode (buffer 8):
//   &2 = in-loop clamp (v34-style)   &4 = no TF   &8 = no comp   &16 = no clamp
// ============================================================================
fragment VolumeFragmentOut fragment_march_phase_batch_w16(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v36mode [[buffer(8)]]) {

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
  const int mode = fc_v36mode;
  const bool noTF   = (mode & 4) != 0;
  const bool noComp = (mode & 8) != 0;
  const bool noClamp = (mode & 16) != 0;
  const bool inLoopClamp = ((mode & 2) != 0) && !noClamp;

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
  float tTerminateMax = s.tTerminateMax;

  float currentT = stepSize;
  int maxSteps = max(1, int(ceil((tEnd - stepSize) / stepSize)));

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 currentPoint = rayOrigin + rayDir * (tStart + stepSize);
  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float3 stepVec = rayDir * stepSize;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;
  bool seenInBounds = false;

  const int unrollN = 16;
  int i = 0;
  const int steps = maxSteps;
  for (; i + unrollN <= steps; i += unrollN)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s0  = sampleVolumeScalar(volumeTexture, evalPoint);
    float s1  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 1.0f);
    float s2  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 2.0f);
    float s3  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 3.0f);
    float s4  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 4.0f);
    float s5  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 5.0f);
    float s6  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 6.0f);
    float s7  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 7.0f);
    float s8  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 8.0f);
    float s9  = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 9.0f);
    float s10 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 10.0f);
    float s11 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 11.0f);
    float s12 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 12.0f);
    float s13 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 13.0f);
    float s14 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 14.0f);
    float s15 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 15.0f);

#define TFETCH(k) (noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h) : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s##k) * scalarScale + scalarBias)), 0.5)))
    half4 c0 = TFETCH(0);  half4 c1 = TFETCH(1);  half4 c2 = TFETCH(2);  half4 c3 = TFETCH(3);
    half4 c4 = TFETCH(4);  half4 c5 = TFETCH(5);  half4 c6 = TFETCH(6);  half4 c7 = TFETCH(7);
    half4 c8 = TFETCH(8);  half4 c9 = TFETCH(9);  half4 c10 = TFETCH(10); half4 c11 = TFETCH(11);
    half4 c12 = TFETCH(12); half4 c13 = TFETCH(13); half4 c14 = TFETCH(14); half4 c15 = TFETCH(15);
#undef TFETCH

    if (!noComp) {
#define COMPK(k) { half w##k = 1.0h - accumulatedOpacity; accumulatedColor += w##k * (c##k.rgb * c##k.a); accumulatedOpacity += w##k * c##k.a; }
      COMPK(0)  COMPK(1)  COMPK(2)  COMPK(3)
      COMPK(4)  COMPK(5)  COMPK(6)  COMPK(7)
      COMPK(8)  COMPK(9)  COMPK(10) COMPK(11)
      COMPK(12) COMPK(13) COMPK(14) COMPK(15)
#undef COMPK
    }

    currentPoint += stepVec * 16.0f;
    currentT += stepSize * 16.0f;
    texLocalPos += texStep * 16.0f;
    evalPoint += evalStep * 16.0f;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }
  for (; i < steps; i++)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s = sampleVolumeScalar(volumeTexture, evalPoint);
    half4 c = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
                   : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s) * scalarScale + scalarBias)), 0.5));
    if (!noComp) {
      half w = 1.0h - accumulatedOpacity;
      accumulatedColor += w * (c.rgb * c.a);
      accumulatedOpacity += w * c.a;
    }
    currentPoint += stepVec;
    currentT += stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }

  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// v37/v38: v36's 16-wide batch scaled to 24/32-wide (inline positions).
// fc_v37mode / fc_v38mode (buffer 8):
//   &2 = in-loop clamp   &4 = no TF   &8 = no comp   &16 = no clamp
// v38/v39/v40: v36's inline-position batch scaled to 32/48/64-wide.
// Like v36, each sample address dies at issue so only the scalar results stay
// live; the wider the batch, the more texture fetches are in flight per warp
// (latency hiding). fc_v38mode / fc_v39mode / fc_v40mode (buffer 8):
//   &2 = in-loop clamp   &4 = no TF   &8 = no comp   &16 = no clamp
// v38/v39/v40: v36's inline-position batch scaled to 32/48/64-wide.
// Like v36, each sample address dies at issue so only the scalar results stay
// live; the wider the batch, the more texture fetches are in flight per warp
// (latency hiding). fc_v38mode / fc_v39mode / fc_v40mode (buffer 8):
//   &2 = in-loop clamp   &4 = no TF   &8 = no comp   &16 = no clamp
// v38/v39/v40: v36's inline-position batch scaled to 32/48/64-wide.
// Like v36, each sample address dies at issue so only the scalar results stay
// live; the wider the batch, the more texture fetches are in flight per warp
// (latency hiding). fc_v38mode / fc_v39mode / fc_v40mode (buffer 8):
//   &2 = in-loop clamp   &4 = no TF   &8 = no comp   &16 = no clamp
#define TFETCHW(k) (noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h) : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s##k) * scalarScale + scalarBias)), 0.5)))
#define COMPKW(k) { half w##k = 1.0h - accumulatedOpacity; accumulatedColor += w##k * (c##k.rgb * c##k.a); accumulatedOpacity += w##k * c##k.a; }

// v38: 32-wide
fragment VolumeFragmentOut fragment_march_phase_batch_w32(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v38mode [[buffer(8)]]) {

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
  const int mode = fc_v38mode;
  const bool noTF   = (mode & 4) != 0;
  const bool noComp = (mode & 8) != 0;
  const bool noClamp = (mode & 16) != 0;
  const bool inLoopClamp = ((mode & 2) != 0) && !noClamp;

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
  float tTerminateMax = s.tTerminateMax;

  float currentT = stepSize;
  int maxSteps = max(1, int(ceil((tEnd - stepSize) / stepSize)));

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 currentPoint = rayOrigin + rayDir * (tStart + stepSize);
  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float3 stepVec = rayDir * stepSize;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;
  bool seenInBounds = false;

  const int unrollN = 32;
  int i = 0;
  const int steps = maxSteps;
  for (; i + unrollN <= steps; i += unrollN)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s0  = sampleVolumeScalar(volumeTexture, evalPoint);
    float s1 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 1.0f);
    float s2 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 2.0f);
    float s3 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 3.0f);
    float s4 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 4.0f);
    float s5 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 5.0f);
    float s6 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 6.0f);
    float s7 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 7.0f);
    float s8 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 8.0f);
    float s9 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 9.0f);
    float s10 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 10.0f);
    float s11 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 11.0f);
    float s12 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 12.0f);
    float s13 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 13.0f);
    float s14 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 14.0f);
    float s15 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 15.0f);
    float s16 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 16.0f);
    float s17 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 17.0f);
    float s18 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 18.0f);
    float s19 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 19.0f);
    float s20 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 20.0f);
    float s21 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 21.0f);
    float s22 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 22.0f);
    float s23 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 23.0f);
    float s24 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 24.0f);
    float s25 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 25.0f);
    float s26 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 26.0f);
    float s27 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 27.0f);
    float s28 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 28.0f);
    float s29 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 29.0f);
    float s30 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 30.0f);
    float s31 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 31.0f);
    half4 c0 = TFETCHW(0); half4 c1 = TFETCHW(1); half4 c2 = TFETCHW(2); half4 c3 = TFETCHW(3);
    half4 c4 = TFETCHW(4); half4 c5 = TFETCHW(5); half4 c6 = TFETCHW(6); half4 c7 = TFETCHW(7);
    half4 c8 = TFETCHW(8); half4 c9 = TFETCHW(9); half4 c10 = TFETCHW(10); half4 c11 = TFETCHW(11);
    half4 c12 = TFETCHW(12); half4 c13 = TFETCHW(13); half4 c14 = TFETCHW(14); half4 c15 = TFETCHW(15);
    half4 c16 = TFETCHW(16); half4 c17 = TFETCHW(17); half4 c18 = TFETCHW(18); half4 c19 = TFETCHW(19);
    half4 c20 = TFETCHW(20); half4 c21 = TFETCHW(21); half4 c22 = TFETCHW(22); half4 c23 = TFETCHW(23);
    half4 c24 = TFETCHW(24); half4 c25 = TFETCHW(25); half4 c26 = TFETCHW(26); half4 c27 = TFETCHW(27);
    half4 c28 = TFETCHW(28); half4 c29 = TFETCHW(29); half4 c30 = TFETCHW(30); half4 c31 = TFETCHW(31);

    if (!noComp) {
      COMPKW(0) COMPKW(1) COMPKW(2) COMPKW(3) COMPKW(4) COMPKW(5) COMPKW(6) COMPKW(7)
      COMPKW(8) COMPKW(9) COMPKW(10) COMPKW(11) COMPKW(12) COMPKW(13) COMPKW(14) COMPKW(15)
      COMPKW(16) COMPKW(17) COMPKW(18) COMPKW(19) COMPKW(20) COMPKW(21) COMPKW(22) COMPKW(23)
      COMPKW(24) COMPKW(25) COMPKW(26) COMPKW(27) COMPKW(28) COMPKW(29) COMPKW(30) COMPKW(31)
    }
    currentPoint += stepVec * 32.0f;
    currentT += stepSize * 32.0f;
    texLocalPos += texStep * 32.0f;
    evalPoint += evalStep * 32.0f;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }
  for (; i < steps; i++) {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s = sampleVolumeScalar(volumeTexture, evalPoint);
    half4 c = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
        : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s) * scalarScale + scalarBias)), 0.5));
    if (!noComp) { half w = 1.0h - accumulatedOpacity; accumulatedColor += w * (c.rgb * c.a); accumulatedOpacity += w * c.a; }
    currentPoint += stepVec; currentT += stepSize; texLocalPos += texStep; evalPoint += evalStep;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }
  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// v39: 48-wide
fragment VolumeFragmentOut fragment_march_phase_batch_w48(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v39mode [[buffer(8)]]) {

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
  const int mode = fc_v39mode;
  const bool noTF   = (mode & 4) != 0;
  const bool noComp = (mode & 8) != 0;
  const bool noClamp = (mode & 16) != 0;
  const bool inLoopClamp = ((mode & 2) != 0) && !noClamp;

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
  float tTerminateMax = s.tTerminateMax;

  float currentT = stepSize;
  int maxSteps = max(1, int(ceil((tEnd - stepSize) / stepSize)));

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 currentPoint = rayOrigin + rayDir * (tStart + stepSize);
  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float3 stepVec = rayDir * stepSize;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;
  bool seenInBounds = false;

  const int unrollN = 48;
  int i = 0;
  const int steps = maxSteps;
  for (; i + unrollN <= steps; i += unrollN)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s0  = sampleVolumeScalar(volumeTexture, evalPoint);
    float s1 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 1.0f);
    float s2 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 2.0f);
    float s3 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 3.0f);
    float s4 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 4.0f);
    float s5 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 5.0f);
    float s6 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 6.0f);
    float s7 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 7.0f);
    float s8 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 8.0f);
    float s9 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 9.0f);
    float s10 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 10.0f);
    float s11 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 11.0f);
    float s12 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 12.0f);
    float s13 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 13.0f);
    float s14 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 14.0f);
    float s15 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 15.0f);
    float s16 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 16.0f);
    float s17 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 17.0f);
    float s18 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 18.0f);
    float s19 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 19.0f);
    float s20 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 20.0f);
    float s21 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 21.0f);
    float s22 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 22.0f);
    float s23 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 23.0f);
    float s24 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 24.0f);
    float s25 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 25.0f);
    float s26 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 26.0f);
    float s27 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 27.0f);
    float s28 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 28.0f);
    float s29 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 29.0f);
    float s30 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 30.0f);
    float s31 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 31.0f);
    float s32 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 32.0f);
    float s33 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 33.0f);
    float s34 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 34.0f);
    float s35 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 35.0f);
    float s36 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 36.0f);
    float s37 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 37.0f);
    float s38 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 38.0f);
    float s39 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 39.0f);
    float s40 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 40.0f);
    float s41 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 41.0f);
    float s42 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 42.0f);
    float s43 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 43.0f);
    float s44 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 44.0f);
    float s45 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 45.0f);
    float s46 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 46.0f);
    float s47 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 47.0f);
    half4 c0 = TFETCHW(0); half4 c1 = TFETCHW(1); half4 c2 = TFETCHW(2); half4 c3 = TFETCHW(3);
    half4 c4 = TFETCHW(4); half4 c5 = TFETCHW(5); half4 c6 = TFETCHW(6); half4 c7 = TFETCHW(7);
    half4 c8 = TFETCHW(8); half4 c9 = TFETCHW(9); half4 c10 = TFETCHW(10); half4 c11 = TFETCHW(11);
    half4 c12 = TFETCHW(12); half4 c13 = TFETCHW(13); half4 c14 = TFETCHW(14); half4 c15 = TFETCHW(15);
    half4 c16 = TFETCHW(16); half4 c17 = TFETCHW(17); half4 c18 = TFETCHW(18); half4 c19 = TFETCHW(19);
    half4 c20 = TFETCHW(20); half4 c21 = TFETCHW(21); half4 c22 = TFETCHW(22); half4 c23 = TFETCHW(23);
    half4 c24 = TFETCHW(24); half4 c25 = TFETCHW(25); half4 c26 = TFETCHW(26); half4 c27 = TFETCHW(27);
    half4 c28 = TFETCHW(28); half4 c29 = TFETCHW(29); half4 c30 = TFETCHW(30); half4 c31 = TFETCHW(31);
    half4 c32 = TFETCHW(32); half4 c33 = TFETCHW(33); half4 c34 = TFETCHW(34); half4 c35 = TFETCHW(35);
    half4 c36 = TFETCHW(36); half4 c37 = TFETCHW(37); half4 c38 = TFETCHW(38); half4 c39 = TFETCHW(39);
    half4 c40 = TFETCHW(40); half4 c41 = TFETCHW(41); half4 c42 = TFETCHW(42); half4 c43 = TFETCHW(43);
    half4 c44 = TFETCHW(44); half4 c45 = TFETCHW(45); half4 c46 = TFETCHW(46); half4 c47 = TFETCHW(47);

    if (!noComp) {
      COMPKW(0) COMPKW(1) COMPKW(2) COMPKW(3) COMPKW(4) COMPKW(5) COMPKW(6) COMPKW(7)
      COMPKW(8) COMPKW(9) COMPKW(10) COMPKW(11) COMPKW(12) COMPKW(13) COMPKW(14) COMPKW(15)
      COMPKW(16) COMPKW(17) COMPKW(18) COMPKW(19) COMPKW(20) COMPKW(21) COMPKW(22) COMPKW(23)
      COMPKW(24) COMPKW(25) COMPKW(26) COMPKW(27) COMPKW(28) COMPKW(29) COMPKW(30) COMPKW(31)
      COMPKW(32) COMPKW(33) COMPKW(34) COMPKW(35) COMPKW(36) COMPKW(37) COMPKW(38) COMPKW(39)
      COMPKW(40) COMPKW(41) COMPKW(42) COMPKW(43) COMPKW(44) COMPKW(45) COMPKW(46) COMPKW(47)
    }
    currentPoint += stepVec * 48.0f;
    currentT += stepSize * 48.0f;
    texLocalPos += texStep * 48.0f;
    evalPoint += evalStep * 48.0f;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }
  for (; i < steps; i++) {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s = sampleVolumeScalar(volumeTexture, evalPoint);
    half4 c = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
        : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s) * scalarScale + scalarBias)), 0.5));
    if (!noComp) { half w = 1.0h - accumulatedOpacity; accumulatedColor += w * (c.rgb * c.a); accumulatedOpacity += w * c.a; }
    currentPoint += stepVec; currentT += stepSize; texLocalPos += texStep; evalPoint += evalStep;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }
  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

// v40: 64-wide
fragment VolumeFragmentOut fragment_march_phase_batch_w64(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    constant int& fixedIterCount [[buffer(3)]],
    constant int& fc_v40mode [[buffer(8)]]) {

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
  const int mode = fc_v40mode;
  const bool noTF   = (mode & 4) != 0;
  const bool noComp = (mode & 8) != 0;
  const bool noClamp = (mode & 16) != 0;
  const bool inLoopClamp = ((mode & 2) != 0) && !noClamp;

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
  float tTerminateMax = s.tTerminateMax;

  float currentT = stepSize;
  int maxSteps = max(1, int(ceil((tEnd - stepSize) / stepSize)));

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  float3 currentPoint = rayOrigin + rayDir * (tStart + stepSize);
  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float3 stepVec = rayDir * stepSize;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;
  bool seenInBounds = false;

  const int unrollN = 64;
  int i = 0;
  const int steps = maxSteps;
  for (; i + unrollN <= steps; i += unrollN)
  {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s0  = sampleVolumeScalar(volumeTexture, evalPoint);
    float s1 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 1.0f);
    float s2 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 2.0f);
    float s3 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 3.0f);
    float s4 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 4.0f);
    float s5 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 5.0f);
    float s6 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 6.0f);
    float s7 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 7.0f);
    float s8 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 8.0f);
    float s9 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 9.0f);
    float s10 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 10.0f);
    float s11 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 11.0f);
    float s12 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 12.0f);
    float s13 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 13.0f);
    float s14 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 14.0f);
    float s15 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 15.0f);
    float s16 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 16.0f);
    float s17 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 17.0f);
    float s18 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 18.0f);
    float s19 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 19.0f);
    float s20 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 20.0f);
    float s21 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 21.0f);
    float s22 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 22.0f);
    float s23 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 23.0f);
    float s24 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 24.0f);
    float s25 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 25.0f);
    float s26 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 26.0f);
    float s27 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 27.0f);
    float s28 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 28.0f);
    float s29 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 29.0f);
    float s30 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 30.0f);
    float s31 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 31.0f);
    float s32 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 32.0f);
    float s33 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 33.0f);
    float s34 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 34.0f);
    float s35 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 35.0f);
    float s36 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 36.0f);
    float s37 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 37.0f);
    float s38 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 38.0f);
    float s39 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 39.0f);
    float s40 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 40.0f);
    float s41 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 41.0f);
    float s42 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 42.0f);
    float s43 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 43.0f);
    float s44 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 44.0f);
    float s45 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 45.0f);
    float s46 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 46.0f);
    float s47 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 47.0f);
    float s48 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 48.0f);
    float s49 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 49.0f);
    float s50 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 50.0f);
    float s51 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 51.0f);
    float s52 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 52.0f);
    float s53 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 53.0f);
    float s54 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 54.0f);
    float s55 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 55.0f);
    float s56 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 56.0f);
    float s57 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 57.0f);
    float s58 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 58.0f);
    float s59 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 59.0f);
    float s60 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 60.0f);
    float s61 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 61.0f);
    float s62 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 62.0f);
    float s63 = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * 63.0f);
    half4 c0 = TFETCHW(0); half4 c1 = TFETCHW(1); half4 c2 = TFETCHW(2); half4 c3 = TFETCHW(3);
    half4 c4 = TFETCHW(4); half4 c5 = TFETCHW(5); half4 c6 = TFETCHW(6); half4 c7 = TFETCHW(7);
    half4 c8 = TFETCHW(8); half4 c9 = TFETCHW(9); half4 c10 = TFETCHW(10); half4 c11 = TFETCHW(11);
    half4 c12 = TFETCHW(12); half4 c13 = TFETCHW(13); half4 c14 = TFETCHW(14); half4 c15 = TFETCHW(15);
    half4 c16 = TFETCHW(16); half4 c17 = TFETCHW(17); half4 c18 = TFETCHW(18); half4 c19 = TFETCHW(19);
    half4 c20 = TFETCHW(20); half4 c21 = TFETCHW(21); half4 c22 = TFETCHW(22); half4 c23 = TFETCHW(23);
    half4 c24 = TFETCHW(24); half4 c25 = TFETCHW(25); half4 c26 = TFETCHW(26); half4 c27 = TFETCHW(27);
    half4 c28 = TFETCHW(28); half4 c29 = TFETCHW(29); half4 c30 = TFETCHW(30); half4 c31 = TFETCHW(31);
    half4 c32 = TFETCHW(32); half4 c33 = TFETCHW(33); half4 c34 = TFETCHW(34); half4 c35 = TFETCHW(35);
    half4 c36 = TFETCHW(36); half4 c37 = TFETCHW(37); half4 c38 = TFETCHW(38); half4 c39 = TFETCHW(39);
    half4 c40 = TFETCHW(40); half4 c41 = TFETCHW(41); half4 c42 = TFETCHW(42); half4 c43 = TFETCHW(43);
    half4 c44 = TFETCHW(44); half4 c45 = TFETCHW(45); half4 c46 = TFETCHW(46); half4 c47 = TFETCHW(47);
    half4 c48 = TFETCHW(48); half4 c49 = TFETCHW(49); half4 c50 = TFETCHW(50); half4 c51 = TFETCHW(51);
    half4 c52 = TFETCHW(52); half4 c53 = TFETCHW(53); half4 c54 = TFETCHW(54); half4 c55 = TFETCHW(55);
    half4 c56 = TFETCHW(56); half4 c57 = TFETCHW(57); half4 c58 = TFETCHW(58); half4 c59 = TFETCHW(59);
    half4 c60 = TFETCHW(60); half4 c61 = TFETCHW(61); half4 c62 = TFETCHW(62); half4 c63 = TFETCHW(63);

    if (!noComp) {
      COMPKW(0) COMPKW(1) COMPKW(2) COMPKW(3) COMPKW(4) COMPKW(5) COMPKW(6) COMPKW(7)
      COMPKW(8) COMPKW(9) COMPKW(10) COMPKW(11) COMPKW(12) COMPKW(13) COMPKW(14) COMPKW(15)
      COMPKW(16) COMPKW(17) COMPKW(18) COMPKW(19) COMPKW(20) COMPKW(21) COMPKW(22) COMPKW(23)
      COMPKW(24) COMPKW(25) COMPKW(26) COMPKW(27) COMPKW(28) COMPKW(29) COMPKW(30) COMPKW(31)
      COMPKW(32) COMPKW(33) COMPKW(34) COMPKW(35) COMPKW(36) COMPKW(37) COMPKW(38) COMPKW(39)
      COMPKW(40) COMPKW(41) COMPKW(42) COMPKW(43) COMPKW(44) COMPKW(45) COMPKW(46) COMPKW(47)
      COMPKW(48) COMPKW(49) COMPKW(50) COMPKW(51) COMPKW(52) COMPKW(53) COMPKW(54) COMPKW(55)
      COMPKW(56) COMPKW(57) COMPKW(58) COMPKW(59) COMPKW(60) COMPKW(61) COMPKW(62) COMPKW(63)
    }
    currentPoint += stepVec * 64.0f;
    currentT += stepSize * 64.0f;
    texLocalPos += texStep * 64.0f;
    evalPoint += evalStep * 64.0f;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }
  for (; i < steps; i++) {
    if (currentT >= tEnd - 1e-6f) break;
    if (inLoopClamp) {
      if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
          any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
        if (seenInBounds) { break; }
        texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      } else {
        seenInBounds = true;
      }
    }
    float s = sampleVolumeScalar(volumeTexture, evalPoint);
    half4 c = noTF ? half4(0.5h, 0.5h, 0.5h, 0.6h)
        : sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s) * scalarScale + scalarBias)), 0.5));
    if (!noComp) { half w = 1.0h - accumulatedOpacity; accumulatedColor += w * (c.rgb * c.a); accumulatedOpacity += w * c.a; }
    currentPoint += stepVec; currentT += stepSize; texLocalPos += texStep; evalPoint += evalStep;
    if (accumulatedOpacity > 1.0h - 1.0h / 255.0h) { break; }
    if (currentT >= tTerminateMax) { break; }
  }
  output.color = float4(float3(accumulatedColor), accumulatedOpacity);
  return output;
}

#undef TFETCHW
#undef COMPKW




// v21: clone of fragment_march_linear_fixedN (v7) but clamp the sample
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
