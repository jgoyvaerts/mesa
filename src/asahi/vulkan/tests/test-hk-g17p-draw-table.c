/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_g17p_draw_table.h"

#include <assert.h>
#include <stdio.h>

/* Replay `draws` draws of a base-instance shader, each with its own
 * firstInstance, through the stream's table bookkeeping as
 * hk_flush_gfx_state() does. Draw d also changes other state when
 * other(d). Returns the vertex state publications (each one a VDM State
 * Update, i.e. a vertex drain); *slotted_out counts draws on the table. */
static unsigned
replay(unsigned draws, bool table, unsigned bits, bool (*other)(unsigned),
       unsigned *slotted_out)
{
   struct hk_g17p_draw_table_cursor c = {0};
   uint64_t next_header = 0x1000;
   unsigned published = 0, slotted_draws = 0;
   for (unsigned d = 0; d < draws; d++) {
      /* The stream's first draw publishes everything. */
      const bool other_state = d == 0 || (other && other(d));
      bool slotted = false;
      if (table && hk_g17p_draw_table_slots(&c, other_state)) {
         if (hk_g17p_draw_table_full(&c, bits)) {
            c.header = next_header += 0x1000;
            c.bits = bits;
            c.next = 0;
         }
         c.next++;
         slotted = true;
         slotted_draws++;
      }
      c.last_other = other_state;
      if (hk_g17p_draw_table_republishes(&c, other_state, true, slotted)) {
         published++;
         c.published = slotted ? c.header : 0;
      }
   }
   if (slotted_out)
      *slotted_out = slotted_draws;
   return published;
}

static unsigned
publications(unsigned draws, bool table, unsigned bits)
{
   return replay(draws, table, bits, NULL, NULL);
}

static bool every_draw(unsigned d) { return true; }
static bool every_tenth(unsigned d) { return d % 10 == 0; }

static void
round_trip(unsigned bits)
{
   static uint32_t table[2 * HK_G17P_DRAW_TABLE_SLOTS];
   const int32_t offsets[] = {0, 5, -3, 0x7fffff00, -0x40000000};
   const uint32_t slots = 1u << bits, mask = UINT32_MAX >> bits;
   for (uint32_t s = 0; s < slots; s++) {
      table[2 * s] = (uint32_t)offsets[s % 5];
      table[2 * s + 1] = 1000 + s;
   }
   const uint32_t indices[] = {0, 1, 0xfffe, 0xffff, mask};
   for (uint32_t s = 0; s < slots; s++) {
      for (unsigned i = 0; i < 5; i++) {
         uint32_t vid, inst;
         hk_g17p_draw_table_decode(
            (indices[i] & mask) + hk_g17p_draw_table_encode(s, bits),
            32 - bits, NULL, table, &vid, &inst);
         assert(vid == (indices[i] & mask) + (uint32_t)offsets[s % 5]);
         assert(inst == 1000 + s);
      }
   }
}

int
main(void)
{
   /* The drain: without the table every draw republishes; with it, one
    * publication per 1 << bits draws. */
   assert(publications(500, false, 8) == 500);
   assert(publications(500, true, 8) == 2);
   assert(publications(HK_G17P_DRAW_TABLE_SLOTS, true, 8) == 1);
   assert(publications(HK_G17P_DRAW_TABLE_SLOTS + 1, true, 8) == 2);
   assert(publications(500, true, 6) == 8);

   /* Every draw republishing anyway: only the first takes a slot. A state
    * change every tenth draw: one publication each, the rest continue. */
   unsigned slotted;
   assert(replay(500, true, 8, every_draw, &slotted) == 500 && slotted == 1);
   /* (plus one when the first table fills at draw 256) */
   assert(replay(500, true, 8, every_tenth, &slotted) == 51 && slotted == 500);
   assert(replay(500, false, 8, every_tenth, NULL) == 500);

   /* Other state always republishes; a shader without base parameters
    * never needs to for them. */
   struct hk_g17p_draw_table_cursor c = {.header = 8, .published = 8,
                                         .bits = 8};
   assert(hk_g17p_draw_table_republishes(&c, true, true, true));
   assert(!hk_g17p_draw_table_republishes(&c, false, true, true));
   assert(hk_g17p_draw_table_republishes(&c, false, true, false));
   assert(!hk_g17p_draw_table_republishes(&c, false, false, false));
   c.published = 0;
   assert(hk_g17p_draw_table_republishes(&c, false, true, true));

   /* A narrower draw needs a table of its own width. */
   c.next = 3;
   assert(!hk_g17p_draw_table_full(&c, 8));
   assert(hk_g17p_draw_table_full(&c, 7));
   c.bits = 2;
   c.next = 4;
   assert(hk_g17p_draw_table_full(&c, 8));

   /* Round trip of every slot, as the hardware forms the raw index: index +
    * bias (indexed, 32-bit wrap) or start + i (non-indexed). The decoded
    * index must equal index + vertexOffset, also wrapped. */
   for (unsigned bits = 1; bits <= HK_G17P_DRAW_TABLE_MAX_BITS; bits++)
      round_trip(bits);

   /* Mode 0: the hardware index (bias applied) and the single entry. */
   const uint32_t single[2] = {77, 9};
   uint32_t vid, inst;
   hk_g17p_draw_table_decode(0xfedcba98, 0, single, NULL, &vid, &inst);
   assert(vid == 0xfedcba98 && inst == 9);

   /* Widths: 8 and 16-bit indices and short draws take full tables. */
   assert(hk_g17p_draw_table_bits(true, 1, 1u << 30, true, true, 0, 0, 0) == 8);
   assert(hk_g17p_draw_table_bits(true, 2, 1u << 30, true, true, 0, 0, 0) == 8);
   assert(hk_g17p_draw_table_bits(false, 0, 1u << 24, true, true, 0, 0, 0) == 8);
   assert(hk_g17p_draw_table_bits(false, 0, (1u << 24) + 1, false, false,
                                  0, 0, 0) == 7);
   assert(hk_g17p_draw_table_bits(false, 0, UINT32_MAX, false, false,
                                  0, 0, 0) == 0);

   /* 32-bit indices: as wide as the largest per-vertex binding allows. */
   const uint64_t small[] = {32ull << 24, 12};
   const uint32_t strides[] = {32, 12};
   assert(hk_g17p_draw_table_bits(true, 4, 3, false, false, small, strides,
                                  2) == 8);
   assert(hk_g17p_draw_table_bits(true, 4, 3, false, false, small, strides,
                                  0) == 8);
   const uint64_t big[] = {(32ull << 24) + 32, 12};
   assert(hk_g17p_draw_table_bits(true, 4, 3, false, false, big, strides,
                                  2) == 7);
   const uint64_t huge[] = {32ull << 31, 12};
   assert(hk_g17p_draw_table_bits(true, 4, 3, false, false, huge, strides,
                                  2) == 1);
   const uint64_t whole[] = {(32ull << 31) + 1, 12};
   assert(hk_g17p_draw_table_bits(true, 4, 3, false, false, whole, strides,
                                  2) == 0);
   assert(!hk_g17p_draw_table_bits(true, 4, 3, true, false, small, strides, 2));
   assert(!hk_g17p_draw_table_bits(true, 4, 3, false, true, small, strides, 2));
   const uint32_t zero_stride[] = {0, 12};
   assert(!hk_g17p_draw_table_bits(true, 4, 3, false, false, small,
                                   zero_stride, 2));

   printf("PASS\n");
   return 0;
}
