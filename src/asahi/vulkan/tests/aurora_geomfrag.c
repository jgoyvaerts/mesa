/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Large render passes shaped like a game's shadow cascade and G-buffer pass,
 * to find whether geometry (vertex shading, tiling) or fragment work limits
 * them:
 *
 *   shadow: 4096x4096 D32_SFLOAT, depth only (no fragment shader), optional
 *           depth bias.
 *   gbuf:   3456x2234, RGBA8 + A2B10G10R10 + B10G11R11 colour targets and
 *           D32_SFLOAT_S8_UINT depth, 1 or 2 samples.
 *
 * The geometry is an indexed grid of `tris` triangles with an interleaved
 * 32-byte vertex (position, normal, uv), split into `draws` draws with a push
 * constant update each, covering `cover` x `cover` of the viewport, drawn
 * `layers` times back to front (every layer passes the depth test). `vcost`
 * and `fcost` add ALU loop iterations to the vertex and fragment shaders.
 * `discard=1` gives the shadow pass an alpha-test fragment shader that
 * discards a quarter of the fragments. `cull=1` culls every triangle after
 * vertex shading (front-and-back culling): the vertex work alone.
 * `pcvary=1` changes the push constants of every draw (new root and vertex
 * state per draw, as games that update per-object constants).
 *
 * Each timed pass is one submit; the min/median/max wall time of `reps` passes
 * is printed. Run with HK_FRAME_ANATOMY=<file> HK_FRAME_ANATOMY_EVERY=1 for
 * the firmware's vertex and fragment times of every pass.
 *
 * Usage: aurora-geomfrag shadow|gbuf [tris=N] [draws=N] [cover=F] [layers=N]
 *        [vcost=N] [fcost=N] [samples=1|2] [bias=0|1] [dstore=0|1] [reps=N]
 *        [discard=0|1] [cull=0|1] [pcvary=0|1]
 * Exit status 0 = correct output (depth and colour checked at the centre and
 * in the corner, which only cover=1 draws).
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "aurora_geomfrag_shaders.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

struct pc {
  float offset_scale[4];
  float z;
  uint32_t vcost, fcost;
  uint32_t draw; /* not read by the shaders */
};

static VkDevice dev;
static VkPhysicalDevice pdev;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(pdev, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
}

static VkDeviceMemory bind_memory(VkMemoryRequirements mr,
                                  VkMemoryPropertyFlags flags) {
  VkDeviceMemory mem;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = memory_type(mr.memoryTypeBits, flags)},
     NULL, &mem));
  return mem;
}

static VkBuffer host_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            void **map) {
  VkBuffer buf;
  CHECK(vkCreateBuffer(dev,
                       &(VkBufferCreateInfo){
                          .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                          .size = size, .usage = usage},
                       NULL, &buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buf, &mr);
  VkDeviceMemory mem = bind_memory(mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  CHECK(vkBindBufferMemory(dev, buf, mem, 0));
  CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, map));
  return buf;
}

static VkImage image(VkFormat fmt, uint32_t w, uint32_t h,
                     VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                     VkImageAspectFlags aspect, VkImageView *view) {
  VkImage img;
  CHECK(vkCreateImage(
     dev,
     &(VkImageCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = {w, h, 1},
        .mipLevels = 1, .arrayLayers = 1, .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage},
     NULL, &img));
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, img, &mr);
  CHECK(vkBindImageMemory(
     dev, img, bind_memory(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
  CHECK(vkCreateImageView(
     dev,
     &(VkImageViewCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
        .subresourceRange = {aspect, 0, 1, 0, 1}},
     NULL, view));
  return img;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc < 2 || (strcmp(argv[1], "shadow") && strcmp(argv[1], "gbuf"))) {
    fprintf(stderr, "usage: %s shadow|gbuf [key=value...]\n", argv[0]);
    return 2;
  }
  const bool gbuf = !strcmp(argv[1], "gbuf");
  unsigned tris = 1 << 20, draws = 500, layers = 1, vcost = 0, fcost = 0;
  unsigned samples = 1, bias = 0, dstore = 1, reps = 7, discard = 0, cull = 0;
  unsigned pcvary = 0;
  double cover = 1.0;
  for (int i = 2; i < argc; i++) {
    char *eq = strchr(argv[i], '=');
    if (!eq) {
      fprintf(stderr, "bad argument %s\n", argv[i]);
      return 2;
    }
    *eq = 0;
    const char *k = argv[i];
    unsigned v = strtoul(eq + 1, NULL, 0);
    if (!strcmp(k, "tris")) tris = v;
    else if (!strcmp(k, "draws")) draws = v;
    else if (!strcmp(k, "cover")) cover = atof(eq + 1);
    else if (!strcmp(k, "layers")) layers = v;
    else if (!strcmp(k, "vcost")) vcost = v;
    else if (!strcmp(k, "fcost")) fcost = v;
    else if (!strcmp(k, "samples")) samples = v;
    else if (!strcmp(k, "bias")) bias = v;
    else if (!strcmp(k, "dstore")) dstore = v;
    else if (!strcmp(k, "reps")) reps = v;
    else if (!strcmp(k, "discard")) discard = v;
    else if (!strcmp(k, "cull")) cull = v;
    else if (!strcmp(k, "pcvary")) pcvary = v;
    else {
      fprintf(stderr, "unknown key %s\n", k);
      return 2;
    }
  }
  if (!draws || !layers || reps < 2 || reps > 256 || (samples != 1 && samples != 2) ||
      (!gbuf && samples != 1) || (gbuf && discard)) {
    fprintf(stderr, "bad parameters\n");
    return 2;
  }
  const uint32_t W = gbuf ? 3456 : 4096, H = gbuf ? 2234 : 4096;

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
           .queueCount = 1, .pQueuePriorities = &prio}},
     NULL, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, 0, 0, &q);

  /* Grid of g x g quads; `tris` is rounded down to whole quads. */
  const unsigned g = (unsigned)sqrt(tris / 2.0);
  const unsigned nquads = g * g;
  if (!g || draws > nquads) {
    fprintf(stderr, "bad parameters\n");
    return 2;
  }
  float *vtx;
  VkBuffer vbuf = host_buffer((VkDeviceSize)(g + 1) * (g + 1) * 32,
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, (void **)&vtx);
  for (unsigned y = 0; y <= g; y++) {
    for (unsigned x = 0; x <= g; x++) {
      float *v = vtx + ((size_t)y * (g + 1) + x) * 8;
      v[0] = (float)x / g, v[1] = (float)y / g, v[2] = 0.5f;
      v[3] = 0, v[4] = 0, v[5] = 1;
      v[6] = v[0], v[7] = v[1];
    }
  }
  uint32_t *idx;
  VkBuffer ibuf = host_buffer((VkDeviceSize)nquads * 6 * 4,
                              VK_BUFFER_USAGE_INDEX_BUFFER_BIT, (void **)&idx);
  for (unsigned y = 0; y < g; y++) {
    for (unsigned x = 0; x < g; x++) {
      uint32_t a = y * (g + 1) + x, *o = idx + ((size_t)y * g + x) * 6;
      o[0] = a, o[1] = a + 1, o[2] = a + g + 1;
      o[3] = a + 1, o[4] = a + g + 2, o[5] = a + g + 1;
    }
  }

  const VkSampleCountFlagBits ms = (VkSampleCountFlagBits)samples;
  const VkFormat zfmt = gbuf ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
  const VkImageAspectFlags zaspect =
     VK_IMAGE_ASPECT_DEPTH_BIT | (gbuf ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
  const VkFormat cfmt[3] = {VK_FORMAT_R8G8B8A8_UNORM,
                            VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                            VK_FORMAT_B10G11R11_UFLOAT_PACK32};
  const unsigned ncol = gbuf ? 3 : 0;
  VkImageView zview, cview[3];
  VkImage zimg = image(zfmt, W, H, ms,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_SAMPLED_BIT,
                       zaspect, &zview);
  /* Single-sampled copies for the readback of a multisampled pass. */
  VkImageView zres_view, c0res_view;
  VkImage zres = samples > 1 ? image(zfmt, W, H, VK_SAMPLE_COUNT_1_BIT,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                     zaspect, &zres_view)
                             : zimg;
  VkImage cimg[3] = {0}, c0res = VK_NULL_HANDLE;
  for (unsigned i = 0; i < ncol; i++) {
    cimg[i] = image(cfmt[i], W, H, ms,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, &cview[i]);
  }
  if (gbuf)
    c0res = samples > 1 ? image(cfmt[0], W, H, VK_SAMPLE_COUNT_1_BIT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT, &c0res_view)
                        : cimg[0];

  VkShaderModule vs, fs, dfs;
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(geomfrag_vert), .pCode = geomfrag_vert},
     NULL, &vs));
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(geomfrag_frag), .pCode = geomfrag_frag},
     NULL, &fs));
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(geomfrag_discard_frag),
        .pCode = geomfrag_discard_frag},
     NULL, &dfs));
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(
     dev,
     &(VkPipelineLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &(VkPushConstantRange){
           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
           sizeof(struct pc)}},
     NULL, &pl));
  VkPipelineColorBlendAttachmentState blend[3] = {
     {.colorWriteMask = 0xf}, {.colorWriteMask = 0xf}, {.colorWriteMask = 0xf}};
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipeline pipe;
  CHECK(vkCreateGraphicsPipelines(
     dev, VK_NULL_HANDLE, 1,
     &(VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &(VkPipelineRenderingCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
           .colorAttachmentCount = ncol, .pColorAttachmentFormats = cfmt,
           .depthAttachmentFormat = zfmt,
           .stencilAttachmentFormat = gbuf ? zfmt : VK_FORMAT_UNDEFINED},
        .stageCount = gbuf || discard ? 2 : 1,
        .pStages = (VkPipelineShaderStageCreateInfo[]){
           {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
           {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = gbuf ? fs : dfs,
            .pName = "main"}},
        .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
           .vertexBindingDescriptionCount = 1,
           .pVertexBindingDescriptions = &(VkVertexInputBindingDescription){
              0, 32, VK_VERTEX_INPUT_RATE_VERTEX},
           .vertexAttributeDescriptionCount = 3,
           .pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]){
              {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
              {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
              {2, 0, VK_FORMAT_R32G32_SFLOAT, 24}}},
        .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
           .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
           .viewportCount = 1, .scissorCount = 1},
        .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
           .cullMode = cull ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE,
           .depthBiasEnable = bias, .depthBiasConstantFactor = 1.25f,
           .depthBiasSlopeFactor = 1.75f, .lineWidth = 1},
        .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
           .rasterizationSamples = ms},
        .pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
           .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
           .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL},
        .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
           .attachmentCount = ncol, .pAttachments = blend},
        .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
           .dynamicStateCount = 2, .pDynamicStates = dyn},
        .layout = pl},
     NULL, &pipe));

  VkCommandPool pool;
  CHECK(vkCreateCommandPool(
     dev,
     &(VkCommandPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT},
     NULL, &pool));
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(
     dev,
     &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1},
     &cb));
  VkFence fence;
  CHECK(vkCreateFence(
     dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
     NULL, &fence));

  /* The last of the reps + 1 passes resolves (1x: stores) for the check. */
  const unsigned ntotal = reps + 1;
  double ms_pass[257];
  for (unsigned rep = 0; rep < ntotal; rep++) {
    const bool check = rep == ntotal - 1;
    CHECK(vkBeginCommandBuffer(
       cb, &(VkCommandBufferBeginInfo){
              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
    VkImageMemoryBarrier2 bar[4];
    unsigned nbar = 0;
    const VkImage targets[4] = {zimg, cimg[0], cimg[1], cimg[2]};
    const unsigned ntargets = 1 + ncol;
    for (unsigned i = 0; i < ntargets; i++)
      bar[nbar++] = (VkImageMemoryBarrier2){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = targets[i],
         .subresourceRange = {i ? VK_IMAGE_ASPECT_COLOR_BIT : zaspect, 0, 1, 0, 1}};
    vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
                                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                 .imageMemoryBarrierCount = nbar,
                                 .pImageMemoryBarriers = bar});

    VkRenderingAttachmentInfo catt[3];
    for (unsigned i = 0; i < ncol; i++)
      catt[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = cview[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .resolveMode = check && samples > 1 && i == 0
                           ? VK_RESOLVE_MODE_AVERAGE_BIT
                           : VK_RESOLVE_MODE_NONE,
         .resolveImageView = check && samples > 1 && i == 0 ? c0res_view
                                                            : VK_NULL_HANDLE,
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkRenderingAttachmentInfo zatt = {
       .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
       .imageView = zview, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = dstore || check ? VK_ATTACHMENT_STORE_OP_STORE
                                  : VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .resolveMode = check && samples > 1 ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT
                                           : VK_RESOLVE_MODE_NONE,
       .resolveImageView = check && samples > 1 ? zres_view : VK_NULL_HANDLE,
       .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
       .clearValue = {.depthStencil = {1.0f, 0}}};
    if (check && samples > 1) {
      VkImageMemoryBarrier2 rb[2] = {
         {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
          .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
          .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = zres,
          .subresourceRange = {zaspect, 0, 1, 0, 1}},
         {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
          .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
          .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = c0res,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}};
      vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
                                   .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                   .imageMemoryBarrierCount = 2,
                                   .pImageMemoryBarriers = rb});
    }
    vkCmdBeginRendering(
       cb, &(VkRenderingInfo){
              .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
              .renderArea = {{0, 0}, {W, H}}, .layerCount = 1,
              .colorAttachmentCount = ncol, .pColorAttachments = catt,
              .pDepthAttachment = &zatt,
              .pStencilAttachment = gbuf ? &zatt : NULL});
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdSetViewport(cb, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
    vkCmdSetScissor(cb, 0, 1, &(VkRect2D){{0, 0}, {W, H}});
    vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &(VkDeviceSize){0});
    vkCmdBindIndexBuffer(cb, ibuf, 0, VK_INDEX_TYPE_UINT32);
    /* Centred, back to front: layer l at z offset -0.1 * l / layers. */
    for (unsigned l = 0; l < layers; l++) {
      for (unsigned d = 0; d < draws; d++) {
        struct pc pc = {{-cover, -cover, 2 * cover, 2 * cover},
                        -0.1f * l / layers, vcost, fcost,
                        pcvary ? l * draws + d : 0};
        vkCmdPushConstants(cb, pl,
                           VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        const unsigned q0 = (uint64_t)nquads * d / draws;
        const unsigned q1 = (uint64_t)nquads * (d + 1) / draws;
        vkCmdDrawIndexed(cb, (q1 - q0) * 6, 1, q0 * 6, 0, 0);
      }
    }
    vkCmdEndRendering(cb);
    CHECK(vkEndCommandBuffer(cb));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    CHECK(vkQueueSubmit(q, 1,
                        &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                        .commandBufferCount = 1,
                                        .pCommandBuffers = &cb},
                        fence));
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    CHECK(vkResetFences(dev, 1, &fence));
    /* The first pass (warm-up) and the checked one are not timed. */
    if (rep && !check)
      ms_pass[rep - 1] =
         (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
  }
  const unsigned nt = reps - 1;

  /* Readback: depth (and c0) at the centre and at the corner. */
  void *rbmap;
  VkBuffer rb = host_buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rbmap);
  CHECK(vkBeginCommandBuffer(
     cb, &(VkCommandBufferBeginInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
  vkCmdPipelineBarrier2(
     cb, &(VkDependencyInfo){
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &(VkMemoryBarrier2){
               .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT}});
  const int32_t px[2][2] = {{W / 2, H / 2}, {0, 0}};
  for (unsigned i = 0; i < 2; i++) {
    vkCmdCopyImageToBuffer(
       cb, zres, VK_IMAGE_LAYOUT_GENERAL, rb, 1,
       &(VkBufferImageCopy){i * 4, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                            {px[i][0], px[i][1], 0}, {1, 1, 1}});
    if (gbuf)
      vkCmdCopyImageToBuffer(
         cb, c0res, VK_IMAGE_LAYOUT_GENERAL, rb, 1,
         &(VkBufferImageCopy){16 + i * 4, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              {px[i][0], px[i][1], 0}, {1, 1, 1}});
  }
  CHECK(vkEndCommandBuffer(cb));
  CHECK(vkQueueSubmit(q, 1,
                      &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers = &cb},
                      fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

  const float *z = rbmap;
  const uint32_t *c = (const uint32_t *)((const char *)rbmap + 16);
  /* The front layer is at 0.5 - 0.1 (layers - 1) / layers; depth bias only
   * pushes it back by a small amount. */
  const float zfront = 0.5f - 0.1f * (layers - 1) / layers;
  const bool corner_drawn = cover >= 1.0;
  bool ok = z[0] >= zfront - 1e-6f && z[0] <= zfront + (bias ? 1e-3f : 1e-6f) &&
            (corner_drawn ? z[1] == z[0] : z[1] == 1.0f);
  if (gbuf)
    ok = ok && (c[0] >> 16) == 0xffff && (c[0] & 0xff) - 0x7fu <= 1 &&
         ((c[0] >> 8) & 0xff) - 0x7fu <= 1 && c[1] == (corner_drawn ? c[0] : 0);
  if (cull)
    ok = z[0] == 1.0f && z[1] == 1.0f && (!gbuf || (c[0] == 0 && c[1] == 0));
  if (!ok)
    fprintf(stderr, "readback: z %f %f (expected %f, 1), c0 %08x %08x\n", z[0],
            z[1], zfront, gbuf ? c[0] : 0, gbuf ? c[1] : 0);

  double sorted[256];
  memcpy(sorted, ms_pass, nt * sizeof(double));
  qsort(sorted, nt, sizeof(double), cmp_double);
  printf("%s %ux%u@%u tris=%u draws=%u cover=%.3f layers=%u vcost=%u fcost=%u "
         "bias=%u dstore=%u discard=%u cull=%u pcvary=%u tri_px=%.2f: wall ms min %.3f med %.3f max %.3f "
         "(%u passes)\n",
         argv[1], W, H, samples, nquads * 2, draws, cover, layers, vcost, fcost,
         bias, dstore, discard, cull, pcvary, cover * cover * W * H / (nquads * 2.0), sorted[0],
         sorted[nt / 2], sorted[nt - 1], nt);
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
