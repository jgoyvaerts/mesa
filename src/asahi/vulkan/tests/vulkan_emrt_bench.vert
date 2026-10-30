#version 450
layout(push_constant) uniform PC { float zstep; } pc;
layout(location = 0) flat out int layer;
void main()
{
   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   layer = gl_InstanceIndex;
   gl_Position = vec4(p * 2.0 - 1.0, 1.0 - float(gl_InstanceIndex + 1) * pc.zstep, 1.0);
}
