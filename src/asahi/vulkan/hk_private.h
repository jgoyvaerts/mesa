/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>

#include "vk_log.h"
#include "util/u_debug.h"
#include "asahi/lib/agx_chip.h"
#include "drm-uapi/asahi_drm.h"
#include "vk_util.h"

#define HK_MAX_SETS                      16
#define HK_MAX_PUSH_SIZE                 256
#define HK_MAX_DYNAMIC_BUFFERS           64
#define HK_MAX_RTS                       8
#define HK_MIN_SSBO_ALIGNMENT            16
#define HK_MIN_TEXEL_BUFFER_ALIGNMENT    16
#define HK_MIN_UBO_ALIGNMENT             64
#define HK_MAX_VIEWPORTS                 16
/* The largest per-descriptor stride any binding can take. G17P storage images
 * occupy 80 bytes and round to 96 so every array element's PBE half remains
 * 32-byte aligned. Mutable descriptors must be large enough for that member.
 */
#define HK_MAX_DESCRIPTOR_SIZE           96
#define HK_MAX_PUSH_DESCRIPTORS          32
#define HK_MAX_DESCRIPTOR_SET_SIZE       (1u << 30)
#define HK_MAX_INLINE_UNIFORM_BLOCK_SIZE (1u << 16)
#define HK_MAX_DESCRIPTORS               (1 << 20)
#define HK_PUSH_DESCRIPTOR_SET_SIZE                                            \
   (HK_MAX_PUSH_DESCRIPTORS * HK_MAX_DESCRIPTOR_SIZE)
#define HK_SSBO_BOUNDS_CHECK_ALIGNMENT 4
#define HK_MAX_MULTIVIEW_VIEW_COUNT    32

#define HK_SPARSE_ADDR_SPACE_SIZE (1ull << 39)
#define HK_MAX_BUFFER_SIZE        (1ull << 37)
#define HK_MAX_SHARED_SIZE        (32 * 1024)

struct hk_addr_range {
   uint64_t addr;
   uint64_t range;
};

#define hk_cmd_buffer_device(cmd) ((struct hk_device *)(cmd)->vk.base.device)

#define perf_debug_dev(dev, fmt, ...)                                          \
   do {                                                                        \
      if ((dev)->debug & AGX_DBG_PERF)                                         \
         mesa_log(MESA_LOG_WARN, (MESA_LOG_TAG), (fmt), ##__VA_ARGS__);        \
   } while (0)

#define perf_debug(cmd, fmt, ...)                                              \
   do {                                                                        \
      if (hk_cmd_buffer_device(cmd)->dev.debug & AGX_DBG_PERF)                 \
         mesa_log(MESA_LOG_WARN, (MESA_LOG_TAG), (fmt), ##__VA_ARGS__);        \
   } while (0)

/*
 * Ray tracing needs the kernel's opt-in as well: on the M4 Pro's 25G83 firmware a stalled
 * traversal wedges the whole GPU, so that kernel neither advertises
 * DRM_ASAHI_FEATURE_RENDER_RAY_QUERY nor accepts ray-traversal work unless
 * asahi.t6040_ray_tracing=1.  Without the bit, no ray tracing extension is exposed.
 */
static inline bool
hk_g17p_rt_enabled(enum agx_chip chip, uint64_t kernel_features)
{
   return chip == AGX_CHIP_G17P &&
          (kernel_features & DRM_ASAHI_FEATURE_RENDER_RAY_QUERY) &&
          !debug_get_bool_option("ASAHI_G17P_RT_DISABLE", false);
}
