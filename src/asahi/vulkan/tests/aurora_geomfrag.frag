/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(location = 0) in vec3 nrm;
layout(location = 1) in vec2 uv;
layout(push_constant) uniform PC {
   vec4 offset_scale;
   float z;
   uint vcost;
   uint fcost;
} pc;
layout(location = 0) out vec4 c0;
layout(location = 1) out vec4 c1;
layout(location = 2) out vec4 c2;
void main()
{
   vec4 a = vec4(uv, nrm.xy);
   /* Extra fragment ALU work. */
   for (uint i = 0; i < pc.fcost; i++)
      a = fract(a * 1.37 + a.yzwx * 0.11);
   c0 = vec4(nrm * 0.5 + 0.5, 1.0);
   c1 = a;
   c2 = vec4(uv, a.x, 1.0);
}
