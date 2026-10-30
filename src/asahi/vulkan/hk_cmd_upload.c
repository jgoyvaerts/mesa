/* SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_shader.h"
#include "util/u_cpu_detect.h"
#include "util/u_debug.h"

DEBUG_GET_ONCE_BOOL_OPTION(m5_fragment_root_shadow,
                           "ASAHI_M5_FRAGMENT_ROOT_SHADOW", true)

static struct agx_ptr
hk_pool_root_alloc(struct hk_cmd_buffer *cmd, uint32_t size,
                   uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_native_graphics(&dev->dev))
      return hk_pool_alloc(cmd, size, alignment);

   return hk_pool_alloc_from(cmd, &cmd->uploader.root, size, alignment,
                             HK_CMD_BO_ROOT);
}
/* ASAHI_MESA_DEBUG=perf: which classes of root fields a new upload
 * changes, to tell why roots (and so vertex state publications) churn. */
static void
hk_root_diff_stats(struct hk_cmd_buffer *cmd, const void *old_root,
                   const struct hk_root_descriptor_table *root, size_t head_B,
                   size_t tail_B, size_t root_size)
{
   const uint8_t *a = old_root, *b = (const uint8_t *)root;
   bool push = false, attrib = false, dynamic = false, sets = false,
        other = false;
   for (size_t i = 0; i < root_size; i++) {
      if ((i >= head_B && i < tail_B) || a[i] == b[i] ||
          i < sizeof(root->root_desc_addr))
         continue;
#define IN(field) (i >= hk_root_descriptor_offset(field) && \
                   i < hk_root_descriptor_offset(field) + sizeof(root->field))
      if (IN(push))
         push = true;
      else if (IN(draw.attrib_base) || IN(draw.attrib_clamps) ||
               IN(draw.attrib_strides))
         attrib = true;
      else if (IN(dynamic_buffers))
         dynamic = true;
      else if (IN(sets))
         sets = true;
      else
         other = true;
#undef IN
   }
   hk_upload_stat(cmd, root_diff_push, push);
   hk_upload_stat(cmd, root_diff_attrib, attrib);
   hk_upload_stat(cmd, root_diff_dynamic, dynamic);
   hk_upload_stat(cmd, root_diff_sets, sets);
   hk_upload_stat(cmd, root_diff_other, other);
}


__attribute__((hot)) uint64_t
hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_descriptor_state *desc = hk_get_descriptors_state(cmd, bind_point);
   struct hk_root_descriptor_table *root = &desc->root;

   size_t root_size = sizeof(*root);
   if (dev->dev.chip == AGX_CHIP_G17P &&
       bind_point == VK_PIPELINE_BIND_POINT_COMPUTE) {
      const struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
      if (s && s->root_descriptor_size)
         root_size = MIN2(root_size, s->root_descriptor_size);
   }

   root->printf_buffer_address = dev->dev.printf_buffer_address;
   root->sampler_heap_address =
      dev->samplers.table.bo && dev->samplers.table.bo->va
         ? dev->samplers.table.bo->va->addr
         : 0;
   root->physical_sparse_map = dev->physical_sparse_map
      ? dev->physical_sparse_map->va->addr : 0;
   root->physical_sparse_base = dev->dev.main_heap_base;
   root->physical_sparse_size = dev->dev.main_heap_size;

   /* Dynamic buffers past the highest one ever bound are read by no bound
    * layout. Never read the uninitialized hole in an earlier upload. */
   const size_t head_B = MIN2(root_size,
      hk_root_descriptor_offset(dynamic_buffers) +
      desc->dynamic_buffer_end * sizeof(root->dynamic_buffers[0]));
   const size_t tail_B = hk_root_descriptor_offset(set_dynamic_buffer_start);
   /* CPU-decoded descriptor programs use a new root identity to invalidate
    * their native program caches, even when only set memory changed. */
   if (agx_native_graphics(&dev->dev) && desc->uploaded_root.gpu &&
       desc->uploaded_root_size == root_size &&
       desc->generation == desc->uploaded_root_generation &&
       desc->dynamic_buffer_end == desc->uploaded_dynamic_buffer_end) {
      root->root_desc_addr = desc->uploaded_root.gpu;
      if (memcmp(desc->uploaded_root.cpu, root, head_B) == 0 &&
          (tail_B >= root_size ||
           memcmp((const uint8_t *)desc->uploaded_root.cpu + tail_B,
                  (const uint8_t *)root + tail_B, root_size - tail_B) == 0)) {
         if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
             cmd->state.gfx.root &&
             desc->uploaded_root.gpu != cmd->state.gfx.root)
            cmd->state.gfx.g17p_bind_tail.valid = false;
         if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS)
            cmd->state.gfx.root_cpu = desc->uploaded_root.cpu;
         hk_upload_stat(cmd, root_reuses, 1);
         return desc->uploaded_root.gpu;
      }
   }

   if (unlikely(dev->dev.debug & AGX_DBG_PERF) && desc->uploaded_root.cpu &&
       desc->uploaded_root_size == root_size)
      hk_root_diff_stats(cmd, desc->uploaded_root.cpu, root, head_B, tail_B,
                         root_size);

   struct agx_ptr root_ptr = hk_pool_root_alloc(cmd, root_size, 8);
   if (!root_ptr.gpu)
      return 0;

   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS)
      cmd->state.gfx.root_cpu = root_ptr.cpu;

   /* A HAL200 tail built for the old root names it and must be rebuilt
    * before publication. Byte-identical roots keep that identity. */
   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
       cmd->state.gfx.root && root_ptr.gpu != cmd->state.gfx.root)
      cmd->state.gfx.g17p_bind_tail.valid = false;
   root->root_desc_addr = root_ptr.gpu;
   /* Capture from the CPU source after every publication-time augmentation.
    * Reading the GPU-mapped root back here would pay uncached load latency. */
   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
       agx_chip_g17_m5(dev->dev.chip) &&
       debug_get_option_m5_fragment_root_shadow()) {
      struct hk_graphics_state *gfx = &cmd->state.gfx;
      hk_root_fragment_snapshot_publish(gfx, root_ptr.cpu, root);
   }

   memcpy(root_ptr.cpu, root, head_B);
   if (tail_B < root_size)
      memcpy((uint8_t *)root_ptr.cpu + tail_B, (const uint8_t *)root + tail_B,
             root_size - tail_B);

   /* Cached G17P roots need explicit CPU-to-GPU publication. Clean just this
    * table now; vkEndCommandBuffer completes all recorded root cleans with a
    * single fence before submission can consume them. */
   if (agx_native_graphics(&dev->dev) && util_has_cache_ops()) {
      util_flush_range_no_fence(root_ptr.cpu, head_B);
      if (tail_B < root_size)
         util_flush_range_no_fence((uint8_t *)root_ptr.cpu + tail_B,
                                  root_size - tail_B);
   }

   desc->uploaded_root = root_ptr;
   desc->uploaded_root_size = root_size;
   desc->uploaded_root_generation = desc->generation;
   desc->uploaded_push_generation = desc->push_generation;
   desc->uploaded_dynamic_buffer_end = desc->dynamic_buffer_end;
   hk_upload_stat(cmd, root_uploads, 1);
   hk_upload_stat(cmd, root_bytes, head_B + (tail_B < root_size ? root_size - tail_B : 0));
   return root_ptr.gpu;
}

uint64_t
hk_cmd_upload_constants(struct hk_cmd_buffer *cmd, const struct hk_shader *s,
                        mesa_shader_stage stage)
{
   if (!s || !s->data_size)
      return 0;

   assert(stage < MESA_SHADER_STAGES);
   const bool native = agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev);
   if (native && s->constant_data_id &&
       cmd->state.constant_data[stage].id == s->constant_data_id) {
      hk_upload_stat(cmd, constant_reuses, 1);
      return cmd->state.constant_data[stage].addr;
   }

   uint64_t addr = hk_pool_upload(cmd, s->data_ptr, s->data_size,
                                  HK_MIN_UBO_ALIGNMENT);
   if (!addr)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   else {
      cmd->state.constant_data[stage].id = s->constant_data_id;
      cmd->state.constant_data[stage].addr = addr;
      hk_upload_stat(cmd, constant_uploads, 1);
      hk_upload_stat(cmd, constant_bytes, s->data_size);
   }
   return addr;
}

void
hk_cmd_upload_draw_params(struct hk_cmd_buffer *cmd, uint32_t first,
                          uint32_t first_instance)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   /* A draw-parameter table shader also loads a zero mode word after them
    * (hk_g17p_draw_table.h). [2] is not uploaded otherwise, only compared. */
   const bool table = hk_g17p_draw_table_bound(gfx);
   uint32_t params[] = {first, first_instance, table ? 0 : UINT32_MAX};
   if (agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) &&
       gfx->g17p_draw_params_host_valid && gfx->draw_params &&
       memcmp(gfx->g17p_draw_params_value, params, sizeof(params)) == 0) {
      hk_upload_stat(cmd, draw_param_reuses, 1);
      return;
   }

   const uint64_t addr =
      hk_pool_upload(cmd, params, table ? 12 : 8, 4);
   gfx->g17p_draw_params_host_valid = addr != 0;
   gfx->draw_params = addr;
   if (!addr)
      return;
   memcpy(gfx->g17p_draw_params_value, params, sizeof(params));
   hk_upload_stat(cmd, draw_param_uploads, 1);
   hk_upload_stat(cmd, draw_param_bytes, sizeof(params));
}

void
hk_cmd_upload_draw_id(struct hk_cmd_buffer *cmd, uint32_t draw_id)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) &&
       gfx->g17p_draw_id_host_valid && gfx->draw_id_ptr &&
       gfx->g17p_draw_id_value == draw_id) {
      hk_upload_stat(cmd, draw_id_reuses, 1);
      return;
   }

   gfx->g17p_draw_id_value = draw_id;
   /* Native uniform DMA reads a whole dword, including the high half. */
   gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id, sizeof(draw_id), 4);
   gfx->g17p_draw_id_host_valid = gfx->draw_id_ptr != 0;
   if (!gfx->draw_id_ptr)
      return;
   hk_upload_stat(cmd, draw_id_uploads, 1);
   hk_upload_stat(cmd, draw_id_bytes, sizeof(draw_id));
}

void
hk_cmd_print_upload_stats(struct hk_cmd_buffer *cmd)
{
   const struct hk_upload_stats *s = &cmd->state.upload_stats;
   perf_debug(cmd, "CPU uploads: draws=%" PRIu64
              " descriptor_binds=%" PRIu64 " descriptor_noops=%" PRIu64
              " root=%" PRIu64 "/%" PRIu64 "B root_reuse=%" PRIu64
              " draw_params=%" PRIu64 "/%" PRIu64 "B draw_param_reuse=%" PRIu64
              " draw_id=%" PRIu64 "/%" PRIu64 "B draw_id_reuse=%" PRIu64
              " constants=%" PRIu64 "/%" PRIu64 "B constant_reuse=%" PRIu64
              " root_diff=push:%" PRIu64 ",attrib:%" PRIu64 ",dynamic:%" PRIu64
              ",sets:%" PRIu64 ",other:%" PRIu64
              " vs_publications=%" PRIu64 " draw_table_slots=%" PRIu64
              " unread_push_skips=%" PRIu64,
              s->draws, s->descriptor_binds, s->descriptor_noops,
              s->root_uploads, s->root_bytes, s->root_reuses,
              s->draw_param_uploads, s->draw_param_bytes, s->draw_param_reuses,
              s->draw_id_uploads, s->draw_id_bytes, s->draw_id_reuses,
              s->constant_uploads, s->constant_bytes, s->constant_reuses,
              s->root_diff_push, s->root_diff_attrib, s->root_diff_dynamic,
              s->root_diff_sets, s->root_diff_other, s->vs_publications,
              s->draw_table_slots, s->root_unread_push_skips);
}
