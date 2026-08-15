vertex VolumeVertexOut probeVert(uint vid [[vertex_id]])
{
  float2 uv = float2((vid == 0) ? 0.0 : ((vid == 1) ? 2.0 : 0.0),
                     (vid == 0) ? 0.0 : ((vid == 1) ? 0.0 : 2.0));
  VolumeVertexOut out;
  out.position = float4(uv * 2.0 - 1.0, 0.5, 1.0);
  out.localPos = float3(uv, 0.0);
  out.instanceID = 0;
  return out;
}
