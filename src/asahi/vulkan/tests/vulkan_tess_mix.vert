/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(push_constant) uniform PC { ivec4 rect; uint rgba; float level; uint quad; } pc;
void main()
{
   /* vulkan_tess_batch.vert without tessellation: two triangles covering
    * the rectangle, in clip space of a 1024x1024 viewport. */
   const vec2 tri[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(0, 1),
                              vec2(1, 0), vec2(1, 1), vec2(0, 1));
   vec2 p = mix(vec2(pc.rect.xy), vec2(pc.rect.zw), tri[gl_VertexIndex % 6]);
   gl_Position = vec4(p / 512.0 - 1.0, 0, 1);
}
