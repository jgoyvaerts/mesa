/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Push constants no bound shader reads.
 *
 * Push constants live in the root descriptor table, and a new root makes a
 * G17P draw republish its vertex state, which drains the vertex pipeline.
 * Engines push per-object values on every draw whether or not the bound
 * shaders read them. Each shader records the push dwords it may read
 * (hk_shader_info::agx3_push_read_mask, all ones when not known); a pending
 * root that differs from the last uploaded one only in dwords no bound
 * shader reads does not need to be published yet.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Mask of the push dwords overlapping bytes [start_B, end_B) of the push
 * range (offsets relative to its start), clamped to the 64 dwords a mask
 * holds. */
static inline uint64_t
hk_push_dwords(uint64_t start_B, uint64_t end_B)
{
   uint64_t mask = 0;
   for (uint64_t dw = start_B / 4; dw < 64 && dw * 4 < end_B; dw++)
      mask |= UINT64_C(1) << dw;
   return mask;
}

/* Whether root `cur` differs from `old` in a byte a shader could observe:
 * any byte in [addr_B, size) outside the unused hole [hole_B, hole_end_B),
 * except push bytes [push_B, push_B + 4 * 64) whose dword is not in
 * `read_mask`. The layout is addr_B <= push_B, push_B + 256 <= hole_B <=
 * hole_end_B <= size (the root address comes first, the dynamic buffers
 * after the push constants). */
static inline bool
hk_root_diff_observed(const uint8_t *old, const uint8_t *cur, size_t size,
                      size_t addr_B, size_t hole_B, size_t hole_end_B,
                      size_t push_B, uint64_t read_mask)
{
   const size_t push_end_B = push_B + 4 * 64;
   if (memcmp(old + addr_B, cur + addr_B, push_B - addr_B) ||
       memcmp(old + push_end_B, cur + push_end_B, hole_B - push_end_B) ||
       memcmp(old + hole_end_B, cur + hole_end_B, size - hole_end_B))
      return true;
   for (unsigned dw = 0; read_mask; dw++, read_mask >>= 1) {
      if ((read_mask & 1) &&
          memcmp(old + push_B + 4 * dw, cur + push_B + 4 * dw, 4))
         return true;
   }
   return false;
}
