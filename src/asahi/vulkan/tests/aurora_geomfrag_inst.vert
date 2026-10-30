/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 nrm;
layout(location = 2) in vec2 uv;
/* Per-instance depth offset (instance-rate binding 1). */
layout(location = 3) in float inst_dz;
layout(push_constant) uniform PC {
   vec4 offset_scale;
   float z;
   uint vcost;
   uint fcost;
} pc;
layout(location = 0) out vec3 o_nrm;
layout(location = 1) out vec2 o_uv;
void main()
{
   vec3 p = pos;
   /* Extra vertex ALU work; the loop leaves p unchanged. */
   for (uint i = 0; i < pc.vcost; i++)
      p += sin(p.yzx * 7.0) * 1e-30;
   gl_Position = vec4(p.xy * pc.offset_scale.zw + pc.offset_scale.xy, p.z + pc.z + inst_dz, 1.0);
   o_nrm = nrm;
   o_uv = uv;
}
