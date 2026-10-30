/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_root_push_mask.h"

#include <assert.h>
#include <stdio.h>

/* A root shaped like hk_root_descriptor_table: an 8-byte address, fields,
 * 256 push bytes, dynamic buffers with an unused hole, a tail. */
#define SIZE 800
#define PUSH 64
#define HOLE 400
#define HOLE_END 700

static bool
observed(const uint8_t *a, const uint8_t *b, uint64_t mask)
{
   return hk_root_diff_observed(a, b, SIZE, 8, HOLE, HOLE_END, PUSH, mask);
}

int
main(void)
{
   assert(hk_push_dwords(0, 4) == 1);
   assert(hk_push_dwords(2, 6) == 3);
   assert(hk_push_dwords(28, 32) == 1ull << 7);
   assert(hk_push_dwords(0, 256) == ~0ull);
   assert(hk_push_dwords(0, 1000) == ~0ull);
   assert(hk_push_dwords(4, 4) == 0);

   static uint8_t a[SIZE], b[SIZE];
   assert(!observed(a, b, ~0ull));

   /* The root address and the hole are never observed. */
   b[3] = 1;
   b[HOLE + 5] = 1;
   assert(!observed(a, b, ~0ull));

   /* A push dword counts only if a shader reads it: the pcvary case, where
    * the shaders read dwords 0-6 and the draw changes dword 7. */
   b[PUSH + 28] = 9;
   assert(!observed(a, b, 0x7f));
   assert(observed(a, b, 0xff));
   assert(!observed(a, b, 0));

   /* Any other field is observed whatever the mask. */
   b[PUSH - 1] = 1;
   assert(observed(a, b, 0));
   b[PUSH - 1] = 0;
   b[PUSH + 256] = 1;
   assert(observed(a, b, 0));
   b[PUSH + 256] = 0;
   b[HOLE_END] = 1;
   assert(observed(a, b, 0));

   printf("PASS\n");
   return 0;
}
