/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(location = 1) in vec2 uv;
layout(push_constant) uniform PC {
   vec4 offset_scale;
   float z;
   uint vcost;
   uint fcost;
} pc;
/* Alpha-tested caster: depth only, a quarter of the fragments of each quad
 * discarded (never the quad's own corner, which the readback checks). */
void main()
{
   vec2 f = fract(uv * 64.0);
   for (uint i = 0; i < pc.fcost; i++)
      f = fract(f * 1.37 + f.yx * 0.11);
   if (f.x > 0.5 && f.y > 0.5)
      discard;
}
