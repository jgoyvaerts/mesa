/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include "asahi/lib/agx_chip.h"
#include "libagx_dgc.h"

struct hk_cmd_buffer;
struct hk_cs;

/*
 * Software tessellation runs five compute stages before the render, each
 * reading what the previous one wrote (hk_launch_tess()). Recorded draw by
 * draw, every stage needs a cache flush: a pass of a few hundred terrain
 * patches becomes thousands of serialized tiny dispatches. The draws of one
 * render do not read each other's tessellation state, so their stages can
 * run side by side: each stage is recorded into its own side stream, and
 * hk_tess_batch_flush() splices the streams into the pre-graphics stream
 * with one flush after each stage.
 *
 * Within a side stream, consecutive launches carry no barrier at all, as
 * consecutive application dispatches on G15 (hk_dispatch_g17p_native()).
 * Even the weak ordering barrier that ends every native launch waits for the
 * launch before it: with it, a pass of 1024 draws (vulkan-tess-batch) takes
 * 27 ms on the M4 Pro whether or not the stages also flush; without it, 1 ms.
 *
 * Any other work for the pre-graphics stream flushes the batch first
 * (hk_cmd_buffer_get_cs_general()), as does the end of the render, so that
 * work keeps its place after the tessellation of the draws before it.
 */
enum hk_tess_stage {
   HK_TESS_STAGE_VS,
   HK_TESS_STAGE_TCS,
   HK_TESS_STAGE_COUNT,
   HK_TESS_STAGE_PREFIX_SUM,
   HK_TESS_STAGE_TESSELLATE,
   HK_TESS_STAGES,
};

struct hk_tess_batch {
   struct hk_cs *stages[HK_TESS_STAGES];

   /* Draws recorded into the side streams since the last flush. */
   unsigned draws;

   /* The pre-graphics stream is being fetched for a batched draw, which must
    * not flush the batch. */
   bool recording;
};

/*
 * Whether a draw's tessellation may join the batch. An indirect draw sizes
 * its stages on the GPU with a setup kernel of its own, and a geometry shader
 * consumes the tessellation right away. Pipeline statistics, clipper and
 * primitives-generated counters are incremented without atomics by every
 * draw's stages, so concurrent stages would lose counts.
 *
 * shortcut: G17P only, the one native chip this was measured on; G15/G16
 * native graphics take the same dispatch path and could join once tested.
 */
static inline bool
hk_tess_batchable(enum agx_chip chip, struct agx_draw draw, bool geometry,
                  bool counters)
{
   return chip == AGX_CHIP_G17P && !agx_is_indirect(draw.b) && !geometry &&
          !counters;
}

/* Create the side streams that are missing. False, with the command buffer
 * error set, if they cannot be allocated. */
bool hk_tess_batch_open(struct hk_cmd_buffer *cmd);

/* The side stream of a stage, ready for the next draw's launch: the weak
 * ordering barrier of the launch before it is dropped. */
struct hk_cs *hk_tess_batch_stage(struct hk_cmd_buffer *cmd,
                                  enum hk_tess_stage stage);

/* Splice the batched stages into the pre-graphics stream. */
void hk_tess_batch_flush(struct hk_cmd_buffer *cmd);

/* Drop the side streams of a command buffer being reset or freed. */
void hk_tess_batch_reset(struct hk_cmd_buffer *cmd);
