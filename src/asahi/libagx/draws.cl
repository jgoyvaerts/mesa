/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_abi.h"
#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "agx_pack.h"
#include "libagx_dgc.h"

/*
 * To implement drawIndirectCount generically, we dispatch a kernel to
 * clone-and-patch the indirect buffer, predicating out draws as appropriate.
 */
KERNEL(32)
libagx_predicate_indirect(global uint32_t *out, constant uint32_t *in,
                          constant uint32_t *draw_count, uint32_t stride_el,
                          uint indexed__2)
{
   uint draw = cl_global_id.x;
   uint words = indexed__2 ? 5 : 4;
   bool enabled = draw < *draw_count;
   out += draw * words;
   in += draw * stride_el;

   /* Copy enabled draws, zero predicated draws. */
   for (uint i = 0; i < words; ++i) {
      out[i] = enabled ? in[i] : 0;
   }
}

/* VK_EXT_transform_feedback exposes the byte count written by streamout as a
 * draw source.  Generate the complete VkDrawIndirectCommand on the GPU so a
 * producer earlier in this command buffer never needs CPU readback. */
KERNEL(1)
libagx_draw_indirect_byte_count(global VkDrawIndirectCommand *out,
                                constant uint32_t *byte_count,
                                uint32_t counter_offset,
                                uint32_t vertex_stride,
                                uint32_t instance_count,
                                uint32_t first_instance)
{
   *out = (VkDrawIndirectCommand){
      .vertexCount =
         agx_xfb_vertex_count(*byte_count, counter_offset, vertex_stride),
      .instanceCount = instance_count,
      .firstVertex = 0,
      .firstInstance = first_instance,
   };
}

/*
 * VK_EXT_conditional_rendering predicates both graphics and compute commands
 * with one 32-bit value.  Honeykrisp lowers each affected command to an
 * indirect argument record and runs this kernel immediately before the
 * consumer.  This is deliberately separate from predicate_indirect: Vulkan
 * requires a 32-bit load (D3D12 uses 64-bit predication), permits inversion,
 * and compute dispatch records contain three rather than four/five words.
 */
KERNEL(1)
libagx_conditional_render(global uint32_t *out, constant uint32_t *in,
                          constant uint32_t *predicate, uint32_t words,
                          uint32_t inverted)
{
   bool execute = ((*predicate != 0) != (inverted != 0));

   for (uint i = 0; i < words; ++i)
      out[i] = execute ? in[i] : 0;
}

/*
 * Materialise an adjacency decomposition as a real index buffer.
 *
 * The software-indexing path strips a draw's index/offset information and has
 * the VS prolog remap gl_VertexIndex instead.  That never works on G17P: a
 * hardware vertex shader reads its invocation index from SR 0xdd directly
 * (agx_compile.c, nir_intrinsic_load_vertex_id), so it never observes the
 * AGX_ABI_VIN_VERTEX_ID the prolog exported, and every adjacency draw fetched
 * attributes at in[i] instead of in[map(i)].
 *
 * Writing the mapped -- and, for an indexed draw, already-resolved -- indices
 * into a buffer lets the hardware input assembler perform the remap, which is
 * the one path the vertex stage does observe.  first_vertex is folded in here
 * so the resulting draw needs no index bias of its own.
 */
KERNEL(64)
libagx_write_adj_indices(global uint *out, uint64_t index_buffer,
                         uint index_buffer_range_el, uint count,
                         uint index_size_B, uint first_vertex,
                         enum mesa_prim prim)
{
   uint i = cl_global_id.x;
   if (i >= count)
      return;

   uint src = poly_map_adj_vertex(i, prim);

   out[i] = first_vertex + (index_size_B
                               ? poly_load_index(index_buffer,
                                                 index_buffer_range_el, src,
                                                 index_size_B)
                               : src);
}

/*
 * The indirect form of libagx_write_adj_indices.  The output index count is
 * not known until the GPU has read the indirect command, so the index buffer
 * is allocated from the geometry heap and named by an indexed indirect draw,
 * exactly as the primitive-restart unroll does.
 */
KERNEL(1024)
libagx_write_adj_indices_indirect(global struct poly_heap *heap,
                                  constant uint *in_draw,
                                  global uint *out_draw, uint64_t index_buffer,
                                  uint index_buffer_range_el,
                                  uint index_size_B, enum mesa_prim prim)
{
   uint tid = cl_local_id.x;

   /* VkDrawIndirectCommand and VkDrawIndexedIndirectCommand agree on their
    * first three words; only the tail differs. */
   uint count = libagx_remap_adj_count(in_draw[0], prim);
   uint instances = in_draw[1];
   uint first = in_draw[2];
   uint vertex_offset = index_size_B ? in_draw[3] : 0;
   uint first_instance = index_size_B ? in_draw[4] : in_draw[3];

   local uint offs_sh;
   local uint ok_sh;

   if (tid == 0) {
      uint offs = 0;
      uint ok = count > 0 && poly_heap_try_alloc_offs(heap, count * 4, &offs);

      offs_sh = offs;
      ok_sh = ok;

      /* An exhausted heap draws nothing rather than writing through an
       * invalid address, matching the restart unroll's behaviour. */
      out_draw[0] = ok ? count : 0;
      out_draw[1] = instances;
      out_draw[2] = ok ? (offs / 4) : 0;
      out_draw[3] = 0; /* vertexOffset, folded into the indices below */
      out_draw[4] = first_instance;
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   if (!ok_sh)
      return;

   global uint *out = (global uint *)(heap->base + offs_sh);

   for (uint i = tid; i < count; i += 1024) {
      uint src = poly_map_adj_vertex(i, prim);

      out[i] =
         index_size_B
            ? vertex_offset + poly_load_index(
                                 index_buffer + (uint64_t)first * index_size_B,
                                 index_buffer_range_el, src, index_size_B)
            : first + src;
   }
}

/*
 * Indexing/offseting is in software if necessary so we strip all
 * indexing/offset information.
 */
KERNEL(1)
libagx_draw_without_adj(global VkDrawIndirectCommand *out,
                        global VkDrawIndirectCommand *in,
                        global struct poly_vertex_params *vp,
                        uint64_t index_buffer, uint64_t index_buffer_range_el,
                        int index_size_B, enum mesa_prim prim)
{
   *out = (VkDrawIndirectCommand){
      .vertexCount = libagx_remap_adj_count(in->vertexCount, prim),
      .instanceCount = in->instanceCount,
   };

   /* TODO: Deduplicate */
   if (index_size_B) {
      uint offs = in->firstVertex;

      vp->index_buffer = poly_index_buffer(index_buffer, index_buffer_range_el,
                                           offs, index_size_B);

      vp->index_buffer_range_el =
         poly_index_buffer_range_el(index_buffer_range_el, offs);
   }
}

/* Precondition: len must be < the group size */
static void
libagx_memcpy_small(global uchar *dst, constant uchar *src, uint len, uint tid)
{
   if (tid < len) {
      dst[tid] = src[tid];
   }
}

static void
libagx_memcpy_aligned_uint4(global uint *dst, constant uint *src, uint len,
                            uint tid, uint group_size)
{
   for (uint i = tid; i < len; i += group_size) {
      vstore4(vload4(i, src), i, dst);
   }
}

static void
libagx_memcpy_to_aligned(global uint *dst, constant uchar *src, uint len,
                         uint tid, uint group_size)
{
   /* Copy a few bytes at the start, then continue after them in both */
   global uchar *dst_B = (global uchar *)dst;
   uint start_unaligned = ((uintptr_t)src) & 3;
   if (start_unaligned) {
      uint need = min(4 - start_unaligned, len);
      libagx_memcpy_small(dst_B, src, need, tid);
      dst_B += need;
      src += need;
      len -= need;
   }

   /* Copy a few bytes at the end */
   uint end_unaligned = len & 0xf;
   len -= end_unaligned;
   libagx_memcpy_small(dst_B + len, src + len, end_unaligned, tid);

   /* Now src is word-aligned and len is 16-aligned. dst_B is word-aligned
    * only when src was: an unaligned source leaves it at an odd offset, so
    * copy the body bytewise in that case. */
   if (start_unaligned) {
      for (uint i = tid; i < len; i += group_size)
         dst_B[i] = src[i];
   } else {
      libagx_memcpy_aligned_uint4(dst, (constant uint *)src, len / 16, tid,
                                  group_size);
   }
}

/* Precondition: len must be < the group size */
static void
libagx_memset_small(global uchar *dst, uchar b, int len, uint tid)
{
   if (tid < len) {
      dst[tid] = b;
   }
}

/*
 * AGX does not implement robustBufferAccess2 semantics for
 * index buffers, where out-of-bounds indices read as zero. When we
 * dynamically detect index buffer overread (this if-statement), we need
 * to clone the index buffer and zero-extend it to get robustness.
 *
 * We do this dynamically (generating a VDM draw to consume the result) to avoid
 * expensive allocations & memcpys in the happy path where no out-of-bounds
 * access occurs. Otherwise we could use a hardware indirect draw, rather than
 * generating VDM words directly in shader.
 *
 * TODO: Handle multiple draws in parallel.
 */
KERNEL(32)
libagx_draw_robust_index(global uint32_t *vdm, global struct poly_heap *heap,
                         constant VkDrawIndexedIndirectCommand *cmd,
                         uint64_t in_buf_ptr, uint32_t in_buf_range_B,
                         ushort restart, enum agx_chip chip,
                         enum agx_primitive topology,
                         uint32_t draw_state_context_offset,
                         uint32_t render_state_emitted,
                         uint64_t empty_draw_jump,
                         enum agx_index_size index_size__3,
                         uint32_t restart_index)
{
   uint tid = get_sub_group_local_id();
   bool first = tid == 0;
   enum agx_index_size index_size = index_size__3;

   struct agx_draw draw = agx_draw_indexed(
      cmd->indexCount, cmd->instanceCount, cmd->firstIndex, cmd->vertexOffset,
      cmd->firstInstance, in_buf_ptr, in_buf_range_B, index_size, restart);

   if (agx_indices_to_B(cmd->firstIndex, index_size) >= in_buf_range_B) {
      /* If the entire draw is out-of-bounds, skip it. We handle this specially
       * for both performance and to avoid integer wrapping issues in the main
       * code path (where cmd->firstIndex could get treated as a negative).
       */
      draw.index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw.index_buffer_range_B = 4;
      draw.start = 0;
   } else if (agx_direct_draw_overreads_indices(draw)) {
      uint start_B = agx_indices_to_B(draw.start, draw.index_size);
      constant void *in_buf =
         (constant void *)(draw.index_buffer + start_B);
      uint in_size_el =
         (draw.index_buffer_range_B - start_B) >> draw.index_size;
      uint in_size_B = agx_indices_to_B(in_size_el, index_size);

      /* After a small number of zeroes at the end, extra zeroes cannot change
       * rendering since they will duplicate the same degenerate primitive many
       * times. Therefore we clamp the number of zeroes we need to extend with.
       * This makes the memset constant time.
       */
      draw.b.count[0] =
         agx_robust_index_clamped_count(draw.b.count[0], in_size_el);

      uint out_size_el = draw.b.count[0];
      uint out_size_B = 0, alloc_size_B = 0;
      bool size_valid = agx_robust_index_shadow_sizes(
         out_size_el, index_size, &out_size_B, &alloc_size_B);

      /* Allocate memory for the shadow index buffer.  The general geometry
       * allocator reports overflow through the AGX2 printf/abort ABI.  This
       * G17P-capable precompiled kernel cannot bake a process-local printf
       * address into cached machine code, so make overflow memory-safe here:
       * every lane observes the same atomic allocation and an exhausted heap
       * emits an empty indexed draw instead of touching an invalid address. */
      uint padded_offs = 0;
      if (first && size_valid) {
         padded_offs = atomic_fetch_add(
            (volatile global atomic_uint *)(&heap->bottom), alloc_size_B);
      }

      padded_offs = sub_group_broadcast(padded_offs, 0);
      bool allocated =
         size_valid &&
         poly_heap_allocation_fits(padded_offs, alloc_size_B, heap->size);
      global uchar *padded = heap->base + padded_offs;

      if (!allocated) {
         draw.b.count[0] = 0;
         draw.index_buffer = AGX_ZERO_PAGE_ADDRESS;
         draw.index_buffer_range_B = 4;
         draw.start = 0;
      } else {
         draw.index_buffer = (uintptr_t)padded;
         draw.index_buffer_range_B = out_size_B;
         draw.start = 0;

         /* Clone the index buffer. The destination is aligned as a
          * post-condition of the heap allocation above. */
         libagx_memcpy_to_aligned((global uint *)padded, in_buf, in_size_B,
                                  tid, 32);

         /* Extend with up to 32 zeroes with a small memset */
         libagx_memset_small(padded + in_size_B, 0, out_size_B - in_size_B,
                             tid);
      }
   }

   if (first) {
      agx_vdm_draw_robust_index_restart(vdm, chip, draw, topology,
                                        draw_state_context_offset,
                                        render_state_emitted != 0,
                                        empty_draw_jump, restart_index);
   }
}

/*
 * VK_KHR_ray_tracing_pipeline indirect trace rays.
 *
 * The megakernel reads its launch size out of a VkTraceRaysIndirectCommand2KHR
 * record, and the command stream needs the same launch expressed as
 * threadgroups.  Neither is known on the host for an indirect trace, so one
 * invocation reads the application's three dimensions and publishes both.
 *
 * `out_record_dims` is the address of the `width` member of a record the driver
 * assembled on the host (vkCmdTraceRaysIndirectKHR, whose indirect buffer holds
 * only the dimensions), or zero when the application supplied the whole record
 * itself (vkCmdTraceRaysIndirect2KHR).
 */
KERNEL(1)
libagx_trace_rays_indirect_setup(global uint32_t *out_record_dims,
                                 global VkDispatchIndirectCommand *out_groups,
                                 constant uint32_t *in_dims,
                                 uint32_t local_size_x_log2)
{
   uint32_t width = in_dims[0];
   uint32_t height = in_dims[1];
   uint32_t depth = in_dims[2];

   if (out_record_dims) {
      out_record_dims[0] = width;
      out_record_dims[1] = height;
      out_record_dims[2] = depth;
   }

   /* The megakernel bounds-checks its own launch id, so a partial trailing
    * workgroup retires without tracing.
    *
    * The workgroup size is a power of two the driver owns, so this rounds up
    * with a shift rather than a division: an integer divide by a runtime value
    * is the one arithmetic form most likely to need an encoding this backend
    * does not have, and there is no reason to risk it for a constant. */
   /* Round up WITHOUT the (width + mask) form: that wraps for any width above
    * 0xFFFFFFE0 and silently launches zero workgroups.  maxRayDispatchInvocation-
    * Count is 2^30, so a conforming application cannot reach it -- but the
    * width comes out of a device buffer nothing validates, and a wrap here is
    * a whole trace that quietly does not happen.  The host side computes the
    * same thing the same way (hk_rt_launch_group_count), and
    * test-hk-ray-tracing-core pins the two together at the top of the range;
    * the first version of this kernel used the wrapping form and that test is
    * what caught it. */
   const uint32_t mask = (1u << local_size_x_log2) - 1u;
   *out_groups = (VkDispatchIndirectCommand){
      .x = (width >> local_size_x_log2) + ((width & mask) ? 1u : 0u),
      .y = height,
      .z = depth,
   };
}
