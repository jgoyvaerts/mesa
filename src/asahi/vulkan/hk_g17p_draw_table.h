/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G17P draw-parameter table.
 *
 * Every VDM State Update drains the vertex pipeline: consecutive draws stop
 * overlapping, which costs a run of 10k-triangle draws about 3.5 us per
 * draw on the M4 Pro. A vertex shader that reads the draw's base vertex or
 * base instance (gl_BaseVertex, gl_BaseInstance, gl_InstanceIndex, any
 * instance-rate attribute) had to republish its vertex state, and so drain,
 * for every draw: the parameters reach it as uniforms.
 *
 * Such a shader reads a mode word w in its DRAW_ID uniform (such a shader
 * does not use gl_DrawID):
 *
 *   w == 0: FIRST_VERTEX and BASE_INSTANCE hold {first, first_instance}
 *           and the vertex index is the hardware's (index + bias, or
 *           start + i), as before.
 *   w == 32 - b (b = 1..8): they hold a pointer to a table of
 *           {first, first_instance} entries. The draw carries its slot in
 *           the top b bits of the hardware vertex index (bias or start =
 *           slot << w), and the shader loads its entry and takes index =
 *           raw - (slot << w) + entry.first.
 *
 * The word is the shift and the table never crosses a 4 GiB boundary
 * (HK_G17P_DRAW_TABLE_ALIGN), so that the decode computes nothing from the
 * uniforms alone: such work would move into the vertex shader's constant
 * program, which runs at every vertex state publication.
 *
 * Draws that change nothing else then share one published vertex state, one
 * table slot each, without any record between them. A table has room for
 * 1 << b draws; b is as wide as the draws' raw indices leave room for, at
 * most 8 (HK_G17P_DRAW_TABLE_SLOTS).
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#define HK_G17P_DRAW_TABLE_MAX_BITS 8
#define HK_G17P_DRAW_TABLE_SLOTS (1u << HK_G17P_DRAW_TABLE_MAX_BITS)
/* Table and cells fit in one aligned page: the address's high word is the
 * same for every entry. */
#define HK_G17P_DRAW_TABLE_ALIGN 4096

/* Raw index bits a draw needs: its raw indices are below 1 << bits. */
static inline unsigned
hk_g17p_draw_table_index_bits(uint64_t limit)
{
   unsigned bits = 32 - HK_G17P_DRAW_TABLE_MAX_BITS;
   while (bits < 64 && (UINT64_C(1) << bits) < limit)
      bits++;
   return bits;
}

/* Slot width (1..8) for a direct draw, or 0 if it cannot take a slot. Its
 * raw indices must leave the top bits free: a non-indexed draw's i < count,
 * an 8 or 16-bit index always, a 32-bit index when every per-vertex binding
 * the shader reads holds at most 1 << (32 - b) elements, so that an index
 * at or above that is out of bounds anyway (a zero stride reads one element
 * for every index, so its range bounds nothing). The shader must not use
 * the index itself, and robustness 2 must not demand zeros for such an
 * out-of-bounds vertex: its wrapped index only reads in-bounds data. */
static inline unsigned
hk_g17p_draw_table_bits(bool indexed, unsigned index_size_B, uint32_t count,
                        bool reads_vertex_index, bool robustness2,
                        const uint64_t *ranges, const uint32_t *strides,
                        unsigned bindings)
{
   uint64_t limit;
   if (!indexed) {
      limit = count;
   } else if (index_size_B < 4) {
      limit = UINT64_C(1) << (8 * index_size_B);
   } else {
      if (reads_vertex_index || robustness2)
         return 0;
      limit = 1;
      for (unsigned i = 0; i < bindings; i++) {
         if (!strides[i])
            return 0;
         const uint64_t elements = (ranges[i] + strides[i] - 1) / strides[i];
         limit = elements > limit ? elements : limit;
      }
   }
   const unsigned index_bits = hk_g17p_draw_table_index_bits(limit);
   return index_bits < 32 ? 32 - index_bits : 0;
}

/* Bias (indexed) or start (non-indexed) of a draw in slot `slot` of a table
 * of width `bits`. */
static inline uint32_t
hk_g17p_draw_table_encode(uint32_t slot, unsigned bits)
{
   return slot << (32 - bits);
}

/* The shader's decode (hk_g17p_lower_draw_table()), for tests: the vertex
 * index and base instance of raw hardware index `raw` with mode word `w`
 * and uniforms `params` (the values, or the table's address in `entries`'s
 * stead). */
static inline void
hk_g17p_draw_table_decode(uint32_t raw, uint32_t w, const uint32_t *params,
                          const uint32_t *entries, uint32_t *vertex_index,
                          uint32_t *first_instance)
{
   if (!w) {
      *vertex_index = raw;
      *first_instance = params[1];
      return;
   }
   const uint32_t slot = raw >> w;
   const uint32_t *entry = entries + 2 * slot;
   *vertex_index = raw - (slot << w) + entry[0];
   *first_instance = entry[1];
}

/* Per-stream table cursor (hk_cs::g17p_draw_table). `header` is the GPU
 * address of the table's uniform cells (its address, then its width),
 * `published` the cells the stream's current vertex state reads (0: mode 0
 * or none), `last_other` whether the previous draw republished for another
 * reason than its base parameters. */
struct hk_g17p_draw_table_cursor {
   uint64_t header;
   uint64_t published;
   uint32_t next;
   unsigned bits;
   bool last_other;
};

/* Whether a draw that can take a slot does. One that republishes anyway
 * (other_state) after another such draw takes mode 0: when every draw
 * republishes, the table only adds its load to each one's drained start.
 * Otherwise it starts or continues a run of draws on the table. */
static inline bool
hk_g17p_draw_table_slots(const struct hk_g17p_draw_table_cursor *c,
                         bool other_state)
{
   return !other_state || !c->last_other;
}

/* A draw of width `bits` needs a new table before taking a slot: none yet,
 * a full one, or one with more slots than its indices leave room for. */
static inline bool
hk_g17p_draw_table_full(const struct hk_g17p_draw_table_cursor *c,
                        unsigned bits)
{
   return !c->header || c->bits > bits || c->next == (1u << c->bits);
}

/* Whether a draw republishes its vertex state. `base_params` is a shader
 * that reads base vertex/instance, `slotted` a draw that took a table slot,
 * `other_state` any other reason (dirty state, root, shaders, draw id). */
static inline bool
hk_g17p_draw_table_republishes(const struct hk_g17p_draw_table_cursor *c,
                               bool other_state, bool base_params,
                               bool slotted)
{
   if (other_state)
      return true;
   if (!base_params)
      return false;
   return !slotted || c->published != c->header;
}
