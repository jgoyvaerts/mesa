/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"
#include "vk_query_pool.h"

struct agx_bo;

struct hk_query_pool {
   struct vk_query_pool vk;

   uint64_t query_start;
   uint64_t query_stride;

   struct agx_bo *bo;
   void *bo_map;

   /* For timestamp queries, the kernel-assigned timestamp buffer handle. Unused
    * for all other query types
    */
   uint32_t handle;

   unsigned oq_queries;

   /* Occlusion queries: the queue timeline point of the latest submission
    * that ended each query, 0 if none since the last host reset. */
   struct hk_oq_submit {
      uint32_t syncobj;
      uint64_t point;
   } *oq_submit;
};

struct hk_cmd_buffer;
struct hk_queue;

struct hk_oq_end {
   struct hk_query_pool *pool;
   uint32_t query;
};

/* Record the queue timeline point that the submission of cmd signals. */
void hk_query_note_submit(struct hk_cmd_buffer *cmd, struct hk_queue *queue,
                          uint64_t point);

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)
