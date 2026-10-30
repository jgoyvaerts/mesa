#version 450
layout(constant_id = 0) const bool DISCARD = false;
layout(location = 0) flat in int layer;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
void main()
{
   ivec2 c = ivec2(gl_FragCoord.xy);
   if (DISCARD && ((c.x ^ c.y ^ layer) & 3) == 0)
      discard;
   float a = float((c.x * 7 + c.y * 13 + layer * 5) & 15) / 15.0;
   vec3 v = vec3(float(layer & 7) / 7.0, fract(gl_FragCoord.xy / 64.0));
   o0 = vec4(v, a);
   o1 = vec4(v.zxy, a);
   o2 = vec4(v.yzx, a);
}
