/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(set = 0, binding = 0) readonly buffer Value { uint v; } value;
layout(location = 0) out uint o;
void main()
{
   o = value.v + 1u;
}
