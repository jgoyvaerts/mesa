/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A frame shaped like Transport Fever 3's with terrain tessellation: three
 * depth-only shadow passes on a 4096x4096 array, a 3456x2234 2x main pass
 * with R11G11B10F, A2R10G10B10, RGBA8 and depth targets, and four
 * full-resolution passes after it. The shadow and main passes interleave
 * tessellated and plain draws, the main pass brackets plain draws with
 * occlusion queries, and pipeline barriers separate the passes.
 *
 * Every pass must stay one hardware render: run it with ASAHI_MESA_DEBUG=perf
 * and count the "Submitting VDM" lines of a submission, 8 for a frame. With
 * `split` the main pass is suspended in one command buffer and resumed in a
 * second.
 *
 * It also checks the occlusion query results and prints the time per frame.
 *
 * Usage: aurora-tess-mix [frames] [split]   Exit status 0 = correct output.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

#define MAIN_W 3456
#define MAIN_H 2234
#define SHADOW 4096
#define CASCADES 3
#define SHADOW_DRAWS 366
#define MAIN_DRAWS 759
#define POST_PASSES 4
#define QUERIES 64

struct pc {
  int32_t rect[4];
  uint32_t rgba;
  float level;
  uint32_t quad;
};

static VkDevice dev;
static VkPhysicalDevice pdev;
static VkQueue q;
static VkFence fence;
static VkPipelineLayout pl;
static VkShaderModule vs_tess, vs_plain, tcs, tes, fs;
static VkImage images[8];
static unsigned nimages;

#include "vulkan_tess_batch_shaders.h"
#include "vulkan_tess_mix_shaders.h"

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(pdev, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
}

static VkShaderModule module(const uint32_t *code, size_t size) {
  VkShaderModule m;
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size, .pCode = code},
     NULL, &m));
  return m;
}

static VkImageView image(VkFormat fmt, uint32_t w, uint32_t h, uint32_t layers,
                         VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                         VkImageAspectFlags aspect, VkImage *out,
                         VkImageView *layer_views) {
  VkImage img;
  CHECK(vkCreateImage(
     dev,
     &(VkImageCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = {w, h, 1},
        .mipLevels = 1, .arrayLayers = layers, .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage},
     NULL, &img));
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, img, &mr);
  VkDeviceMemory mem;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex =
           memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
     NULL, &mem));
  CHECK(vkBindImageMemory(dev, img, mem, 0));
  VkImageView view = VK_NULL_HANDLE;
  for (uint32_t l = 0; l < layers; l++) {
    CHECK(vkCreateImageView(
       dev,
       &(VkImageViewCreateInfo){
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
          .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
          .subresourceRange = {aspect, 0, 1, l, 1}},
       NULL, layer_views ? &layer_views[l] : &view));
  }
  if (out)
    *out = img;
  images[nimages++] = img;
  return layer_views ? layer_views[0] : view;
}

/* A pipeline drawing pc.rect, tessellated (triangle domain) or not. */
static VkPipeline pipeline(bool tess, const VkFormat *colours, uint32_t ncol,
                           VkFormat depth, VkSampleCountFlagBits samples) {
  const VkPipelineShaderStageCreateInfo stages[] = {
     {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = tess ? vs_tess : vs_plain,
      .pName = "main"},
     {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
     {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, .module = tcs,
      .pName = "main"},
     {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, .module = tes,
      .pName = "main"},
  };
  VkPipelineColorBlendAttachmentState blend[4];
  for (unsigned i = 0; i < 4; i++)
    blend[i] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf};
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipeline p;
  CHECK(vkCreateGraphicsPipelines(
     dev, VK_NULL_HANDLE, 1,
     &(VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &(VkPipelineRenderingCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
           .colorAttachmentCount = ncol, .pColorAttachmentFormats = colours,
           .depthAttachmentFormat = depth},
        .stageCount = tess ? 4 : 2, .pStages = stages,
        .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
        .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
           .topology = tess ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
                            : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
        .pTessellationState = &(VkPipelineTessellationStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
           .patchControlPoints = 3},
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
           .viewportCount = 1, .scissorCount = 1},
        .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
           .lineWidth = 1},
        .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
           .rasterizationSamples = samples},
        .pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
           .depthTestEnable = depth != VK_FORMAT_UNDEFINED,
           .depthWriteEnable = depth != VK_FORMAT_UNDEFINED,
           .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL},
        .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
           .attachmentCount = ncol, .pAttachments = blend},
        .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
           .dynamicStateCount = 2, .pDynamicStates = dyn},
        .layout = pl},
     NULL, &p));
  return p;
}

static void memory_barrier(VkCommandBuffer cb) {
  vkCmdPipelineBarrier2(
     cb, &(VkDependencyInfo){
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &(VkMemoryBarrier2){
               .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
               .dstAccessMask =
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT}});
}

static void begin(VkCommandBuffer cb, VkRect2D area,
                  const VkRenderingAttachmentInfo *colours, uint32_t ncol,
                  const VkRenderingAttachmentInfo *depth,
                  VkRenderingFlags flags) {
  vkCmdBeginRendering(cb, &(VkRenderingInfo){
                             .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                             .flags = flags, .renderArea = area,
                             .layerCount = 1, .colorAttachmentCount = ncol,
                             .pColorAttachments = colours,
                             .pDepthAttachment = depth});
  vkCmdSetViewport(cb, 0, 1, &(VkViewport){0, 0, 1024, 1024, 0, 1});
  vkCmdSetScissor(cb, 0, 1, &area);
}

/* Draw i of a pass: small rectangles over the 1024x1024 viewport. */
static void draw(VkCommandBuffer cb, VkPipeline p, unsigned i) {
  struct pc c = {
     .rect = {(i % 32) * 32, (i / 32 % 32) * 32, (i % 32) * 32 + 40,
              (i / 32 % 32) * 32 + 40},
     .rgba = 0xff000000u | (i * 2654435761u >> 8), .level = 1 + i % 7,
  };
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(c), &c);
  vkCmdDraw(cb, 6, 1, 0, 0);
}

/* All images from UNDEFINED to GENERAL, once. */
static void to_general(VkCommandBuffer cb) {
  VkImageMemoryBarrier2 b[8];
  for (unsigned i = 0; i < nimages; i++) {
    b[i] = (VkImageMemoryBarrier2){
       .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
       .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
       .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
       .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = images[i],
       .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                            VK_REMAINING_ARRAY_LAYERS}};
    if (i >= 3 && i <= 4)
      b[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
                               .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                               .imageMemoryBarrierCount = nimages,
                               .pImageMemoryBarriers = b});
}

int main(int argc, char **argv) {
  const unsigned frames = argc > 1 ? atoi(argv[1]) : 3;
  const bool split = argc > 2 && !strcmp(argv[2], "split");
  VkInstance inst;
  CHECK(vkCreateInstance(
     &(VkInstanceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){
           .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
           .apiVersion = VK_API_VERSION_1_3}},
     NULL, &inst));
  uint32_t n = 1;
  CHECK(vkEnumeratePhysicalDevices(inst, &n, &pdev));
  if (!n) {
    fprintf(stderr, "no device\n");
    return 2;
  }
  float prio = 1;
  VkPhysicalDeviceVulkan13Features f13 = {
     .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
     .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE};
  CHECK(vkCreateDevice(
     pdev,
     &(VkDeviceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
           .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
           .queueCount = 1, .pQueuePriorities = &prio},
        .pEnabledFeatures = &(VkPhysicalDeviceFeatures){
           .tessellationShader = VK_TRUE}},
     NULL, &dev));
  vkGetDeviceQueue(dev, 0, 0, &q);

  CHECK(vkCreatePipelineLayout(
     dev,
     &(VkPipelineLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &(VkPushConstantRange){
           VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(struct pc)}},
     NULL, &pl));
  vs_tess = module(tess_batch_vert, sizeof(tess_batch_vert));
  vs_plain = module(tess_mix_plain_vert, sizeof(tess_mix_plain_vert));
  tcs = module(tess_batch_tri_tesc, sizeof(tess_batch_tri_tesc));
  tes = module(tess_batch_tri_tese, sizeof(tess_batch_tri_tese));
  fs = module(tess_batch_frag, sizeof(tess_batch_frag));

  const VkFormat main_fmt[3] = {VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                VK_FORMAT_A2R10G10B10_UNORM_PACK32,
                                VK_FORMAT_R8G8B8A8_UNORM};
  const VkFormat zfmt = VK_FORMAT_D32_SFLOAT, post_fmt = VK_FORMAT_R8G8B8A8_UNORM;
  const VkImageUsageFlags col = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT;
  const VkImageUsageFlags zs = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT;

  VkImageView main_views[3], main_z, shadow_views[CASCADES], post_views[2];
  for (unsigned i = 0; i < 3; i++)
    main_views[i] = image(main_fmt[i], MAIN_W, MAIN_H, 1, VK_SAMPLE_COUNT_2_BIT,
                          col, VK_IMAGE_ASPECT_COLOR_BIT, NULL, NULL);
  main_z = image(zfmt, MAIN_W, MAIN_H, 1, VK_SAMPLE_COUNT_2_BIT, zs,
                 VK_IMAGE_ASPECT_DEPTH_BIT, NULL, NULL);
  image(zfmt, SHADOW, SHADOW, CASCADES, VK_SAMPLE_COUNT_1_BIT, zs,
        VK_IMAGE_ASPECT_DEPTH_BIT, NULL, shadow_views);
  for (unsigned i = 0; i < 2; i++)
    post_views[i] = image(post_fmt, MAIN_W, MAIN_H, 1, VK_SAMPLE_COUNT_1_BIT,
                          col, VK_IMAGE_ASPECT_COLOR_BIT, NULL, NULL);

  VkPipeline main_tess = pipeline(true, main_fmt, 3, zfmt, VK_SAMPLE_COUNT_2_BIT);
  VkPipeline main_plain = pipeline(false, main_fmt, 3, zfmt, VK_SAMPLE_COUNT_2_BIT);
  VkPipeline shadow_tess = pipeline(true, NULL, 0, zfmt, VK_SAMPLE_COUNT_1_BIT);
  VkPipeline shadow_plain = pipeline(false, NULL, 0, zfmt, VK_SAMPLE_COUNT_1_BIT);
  VkPipeline post = pipeline(false, &post_fmt, 1, VK_FORMAT_UNDEFINED,
                             VK_SAMPLE_COUNT_1_BIT);

  VkQueryPool queries;
  CHECK(vkCreateQueryPool(
     dev,
     &(VkQueryPoolCreateInfo){.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                              .queryType = VK_QUERY_TYPE_OCCLUSION,
                              .queryCount = QUERIES},
     NULL, &queries));

  VkCommandPool pool;
  CHECK(vkCreateCommandPool(
     dev,
     &(VkCommandPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT},
     NULL, &pool));
  VkCommandBuffer cbs[2];
  CHECK(vkAllocateCommandBuffers(
     dev,
     &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 2},
     cbs));
  CHECK(vkCreateFence(
     dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
     NULL, &fence));

  const VkRect2D main_area = {{0, 0}, {MAIN_W, MAIN_H}};
  VkRenderingAttachmentInfo main_att[3], main_zatt = {
     .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
     .imageView = main_z, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
     .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
     .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
     .clearValue.depthStencil = {1, 0}};
  for (unsigned i = 0; i < 3; i++)
    main_att[i] = (VkRenderingAttachmentInfo){
       .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
       .imageView = main_views[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = VK_ATTACHMENT_STORE_OP_STORE};

  double ms_total = 0, ms_min = 1e9;
  unsigned bad = 0;
  for (unsigned f = 0; f < frames; f++) {
    VkCommandBuffer cb = cbs[0];
    CHECK(vkBeginCommandBuffer(
       cb, &(VkCommandBufferBeginInfo){
              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
    if (!f)
      to_general(cb);
    vkCmdResetQueryPool(cb, queries, 0, QUERIES);
    memory_barrier(cb);

    /* Shadow cascades: every fifth draw is not tessellated. */
    for (unsigned c = 0; c < CASCADES; c++) {
      begin(cb, (VkRect2D){{0, 0}, {SHADOW, SHADOW}}, NULL, 0,
            &(VkRenderingAttachmentInfo){
               .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = shadow_views[c],
               .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
               .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
               .clearValue.depthStencil = {1, 0}},
            0);
      for (unsigned i = 0; i < SHADOW_DRAWS; i++) {
        const bool tess = i % 5 != 4;
        draw(cb, tess ? shadow_tess : shadow_plain, i);
      }
      vkCmdEndRendering(cb);
      memory_barrier(cb);
    }

    /* Main pass: a plain draw after every second tessellated one, and every
     * twelfth plain draw inside an occlusion query. */
    begin(cb, main_area, main_att, 3, &main_zatt,
          split ? VK_RENDERING_SUSPENDING_BIT : 0);
    unsigned query = 0;
    for (unsigned i = 0; i < MAIN_DRAWS; i++) {
      if (split && i == MAIN_DRAWS / 2) {
        vkCmdEndRendering(cb);
        CHECK(vkEndCommandBuffer(cb));
        cb = cbs[1];
        CHECK(vkBeginCommandBuffer(
           cb, &(VkCommandBufferBeginInfo){
                  .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
        begin(cb, main_area, main_att, 3, &main_zatt,
              VK_RENDERING_RESUMING_BIT);
      }
      draw(cb, main_tess, i);
      if (i % 2) {
        const bool counted = i % 24 == 1 && query < QUERIES;
        if (counted)
          vkCmdBeginQuery(cb, queries, query, 0);
        draw(cb, main_plain, i + 1);
        if (counted)
          vkCmdEndQuery(cb, queries, query++);
      }
    }
    vkCmdEndRendering(cb);
    memory_barrier(cb);

    /* Full-resolution passes, each reading the previous one's output. */
    for (unsigned p = 0; p < POST_PASSES; p++) {
      begin(cb, main_area,
            &(VkRenderingAttachmentInfo){
               .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = post_views[p & 1],
               .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
               .storeOp = VK_ATTACHMENT_STORE_OP_STORE},
            1, NULL, 0);
      draw(cb, post, p);
      vkCmdEndRendering(cb);
      memory_barrier(cb);
    }
    CHECK(vkEndCommandBuffer(cb));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    CHECK(vkQueueSubmit(q, 1,
                        &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                        .commandBufferCount = split ? 2 : 1,
                                        .pCommandBuffers = cbs},
                        fence));
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    CHECK(vkResetFences(dev, 1, &fence));
    const double ms =
       (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    ms_total += ms;
    if (ms < ms_min)
      ms_min = ms;

    /* Every counted draw is a visible rectangle at depth 0. */
    uint64_t results[QUERIES];
    CHECK(vkGetQueryPoolResults(dev, queries, 0, query, sizeof(results),
                                results, sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT |
                                   VK_QUERY_RESULT_WAIT_BIT));
    for (unsigned i = 0; i < query; i++) {
      if (!results[i] && bad++ < 8)
        fprintf(stderr, "frame %u: query %u counted no samples\n", f, i);
    }
  }

  printf("%u frames%s: %.2f ms per frame (min %.2f)\n", frames,
         split ? " (main pass split)" : "", ms_total / frames, ms_min);
  if (bad) {
    printf("FAIL: %u empty queries\n", bad);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
