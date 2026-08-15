#include <metal_stdlib>
using namespace metal;

struct VSOut { float4 pos [[position]]; float3 localPos; };
struct U { float4 a; float4 b; float4 c; };
constant sampler sampLinear = sampler(coord::normalized, address::clamp_to_edge, filter::linear);
constant sampler sampNearest = sampler(coord::normalized, address::clamp_to_edge, filter::nearest);

fragment float4 frag_linear(VSOut in [[stage_in]], texture3d<float> tex [[texture(0)]], constant U& u [[buffer(1)]]) {
  float3 p = in.localPos;
  float3 step = u.a.xyz;
  float acc = 0.0;
  for (int i = 0; i < 659; i++) {
    float s = tex.sample(sampLinear, p).r;
    acc = max(acc, s);
    p += step;
  }
  return float4(float3(acc), 1.0);
}

fragment float4 frag_nearest(VSOut in [[stage_in]], texture3d<float> tex [[texture(0)]], constant U& u [[buffer(1)]]) {
  float3 p = in.localPos;
  float3 step = u.a.xyz;
  float acc = 0.0;
  for (int i = 0; i < 659; i++) {
    float s = tex.sample(sampNearest, p).r;
    acc = max(acc, s);
    p += step;
  }
  return float4(float3(acc), 1.0);
}
