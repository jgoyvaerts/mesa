/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_tess_batch.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"

bool
hk_tess_batch_open(struct hk_cmd_buffer *cmd)
{
   struct hk_tess_batch *batch = &cmd->tess_batch;

   for (unsigned i = 0; i < HK_TESS_STAGES; ++i) {
      if (!batch->stages[i])
         batch->stages[i] = hk_cs_create(cmd, true);

      if (!batch->stages[i]) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
   }

   return true;
}

struct hk_cs *
hk_tess_batch_stage(struct hk_cmd_buffer *cmd, enum hk_tess_stage stage)
{
   struct hk_cs *cs = cmd->tess_batch.stages[stage];

   /* Before the tail goes: a chunk link must still follow a barrier. */
   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   if (cs->stats.cmds && cs->current == cs->weak_barrier_end) {
      cs->current = (uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE;
      cs->weak_barrier_end = NULL;
   }

   return cs;
}

void
hk_tess_batch_flush(struct hk_cmd_buffer *cmd)
{
   struct hk_tess_batch *batch = &cmd->tess_batch;
   if (!batch->draws)
      return;

   /* Created by the first batched draw, and kept open until after this. */
   struct hk_cs *cs = cmd->current_cs.pre_gfx;
   batch->draws = 0;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   for (unsigned i = 0; i < HK_TESS_STAGES; ++i) {
      struct hk_cs *stage = batch->stages[i];
      batch->stages[i] = NULL;

      /* An empty stream has no barrier at its end to take over. */
      if (cs && stage->stats.cmds) {
         hk_cs_merge_cdm(cs, stage);

         /* The next stage reads what this one wrote. */
         hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
         hk_cdm_cache_flush(dev, cs);
      }

      hk_cs_destroy(stage);
   }
}

void
hk_tess_batch_reset(struct hk_cmd_buffer *cmd)
{
   struct hk_tess_batch *batch = &cmd->tess_batch;

   for (unsigned i = 0; i < HK_TESS_STAGES; ++i) {
      if (batch->stages[i])
         hk_cs_destroy(batch->stages[i]);
   }

   *batch = (struct hk_tess_batch){0};
}
