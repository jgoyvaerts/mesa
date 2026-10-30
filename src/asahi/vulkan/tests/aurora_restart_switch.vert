/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
/* Vertex v sits at grid point (v & 0xff, (v >> 8) & 0xff) of a 256x256 image. */
void main()
{
   uint v = uint(gl_VertexIndex);
   vec2 p = vec2(v & 0xffu, (v >> 8) & 0xffu) / 255.0;
   gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
