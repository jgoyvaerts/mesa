/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Fixed cost of a render pass: N tiny passes (one scissored 32x32 draw or
 * none) on small and full-screen attachments, with each load op, recorded in
 * one command buffer, in N command buffers of one submit, or in N submits.
 * Consecutive passes alternate between two images so that none continues the
 * previous one, except in the last case, which shows what continuing saves.
 *
 * Prints the wall-clock time per pass of every case (submit to fence, N =
 * 400, minus the same submit with N = 1). Cases are 30 ms apart, so a trace
 * of asahi_g17_complete (trace_clock mono) taken around the run splits into
 * one group of jobs per run, by the CLOCK_MONOTONIC window of each N-pass
 * submit printed to stderr ("T <start ms> <end ms> <case>"), for the GPU time
 * per job and the gaps between jobs.
 *
 * Usage: aurora-render-overhead [reps]   Exit status 0 = correct output.
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

#define N 400
#define FMT VK_FORMAT_R8G8B8A8_UNORM

static VkDevice dev;
static VkPhysicalDevice pdev;
static VkQueue q;
static VkCommandPool pool;
static VkCommandBuffer cbs[N];
static VkFence fence;
static VkPipelineLayout pl;
static VkPipeline pipe;

/* Full-screen triangle, constant colour from a push constant. */
#include "vulkan_render_area_shaders.h"

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(pdev, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
}

struct target {
  uint32_t w, h;
  VkImage img[2];
  VkImageView view[2];
};

static void make_target(struct target *t, uint32_t w, uint32_t h) {
  t->w = w;
  t->h = h;
  for (unsigned i = 0; i < 2; i++) {
    CHECK(vkCreateImage(
       dev,
       &(VkImageCreateInfo){
          .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
          .imageType = VK_IMAGE_TYPE_2D, .format = FMT, .extent = {w, h, 1},
          .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
          .tiling = VK_IMAGE_TILING_OPTIMAL,
          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
       NULL, &t->img[i]));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, t->img[i], &mr);
    VkDeviceMemory m;
    CHECK(vkAllocateMemory(
       dev,
       &(VkMemoryAllocateInfo){
          .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
          .allocationSize = mr.size,
          .memoryTypeIndex = memory_type(mr.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
       NULL, &m));
    CHECK(vkBindImageMemory(dev, t->img[i], m, 0));
    CHECK(vkCreateImageView(
       dev,
       &(VkImageViewCreateInfo){
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          .image = t->img[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FMT,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
       NULL, &t->view[i]));
  }
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from,
                    VkImageLayout to) {
  vkCmdPipelineBarrier2(
     cb, &(VkDependencyInfo){
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &(VkImageMemoryBarrier2){
               .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
               .dstAccessMask =
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
               .oldLayout = from, .newLayout = to, .image = img,
               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}});
}

static const uint32_t colour = 0xff20c040u;

/* One pass over `area` of t's image i with `draws` scissored 32x32 draws. */
static void pass(VkCommandBuffer cb, const struct target *t, unsigned i,
                 VkRect2D area, VkAttachmentLoadOp load, unsigned draws) {
  VkClearValue cv = {.color.float32 = {0.25f, 0.5f, 0.75f, 1}};
  vkCmdBeginRendering(
     cb, &(VkRenderingInfo){
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = area,
            .layerCount = 1, .colorAttachmentCount = 1,
            .pColorAttachments = &(VkRenderingAttachmentInfo){
               .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = t->view[i],
               .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               .loadOp = load, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
               .clearValue = cv}});
  for (unsigned d = 0; d < draws; d++) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdSetViewport(cb, 0, 1, &(VkViewport){0, 0, t->w, t->h, 0, 1});
    vkCmdSetScissor(cb, 0, 1,
                    &(VkRect2D){area.offset, {32, 32}});
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &colour);
    vkCmdDraw(cb, 3, 1, 0, 0);
  }
  vkCmdEndRendering(cb);
}

static void begin(VkCommandBuffer cb) {
  CHECK(vkBeginCommandBuffer(
     cb, &(VkCommandBufferBeginInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}));
}

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void sleep_ms(unsigned ms) {
  nanosleep(&(struct timespec){0, ms * 1000000L}, NULL);
}

enum split { ONE_CB, MANY_CB, MANY_SUBMIT };

struct bench {
  const char *name;
  const struct target *t;
  VkRect2D area; /* extent 0 = whole attachment */
  VkAttachmentLoadOp load;
  unsigned draws;
  enum split split;
  bool same; /* all passes on one image: they may continue each other */
};

/* Records and runs n passes as `b` asks, returns submit-to-fence ms. */
static double run(const struct bench *b, unsigned n) {
  VkRect2D area = b->area;
  if (!area.extent.width)
    area.extent = (VkExtent2D){b->t->w, b->t->h};
  const unsigned ncb = b->split == ONE_CB ? 1 : n;
  for (unsigned c = 0; c < ncb; c++) {
    CHECK(vkResetCommandBuffer(cbs[c], 0));
    begin(cbs[c]);
  }
  for (unsigned i = 0; i < n; i++)
    pass(cbs[b->split == ONE_CB ? 0 : i], b->t, b->same ? 0 : i & 1, area, b->load,
         b->draws);
  for (unsigned c = 0; c < ncb; c++)
    CHECK(vkEndCommandBuffer(cbs[c]));

  double t0 = now_ms();
  if (b->split == MANY_SUBMIT) {
    for (unsigned c = 0; c < ncb; c++)
      CHECK(vkQueueSubmit(
         q, 1,
         &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                         .commandBufferCount = 1, .pCommandBuffers = &cbs[c]},
         c == ncb - 1 ? fence : VK_NULL_HANDLE));
  } else {
    CHECK(vkQueueSubmit(
       q, 1,
       &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = ncb, .pCommandBuffers = cbs},
       fence));
  }
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  double t1 = now_ms();
  CHECK(vkResetFences(dev, 1, &fence));
  if (n > 1)
    fprintf(stderr, "T %.3f %.3f %s\n", t0, t1, b->name);
  return t1 - t0;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  const unsigned reps = argc > 1 ? atoi(argv[1]) : 5;
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
  vkGetDeviceQueue(dev, 0, 0, &q);

  VkShaderModule vs, fs;
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(render_area_vert), .pCode = render_area_vert},
     NULL, &vs));
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(render_area_frag), .pCode = render_area_frag},
     NULL, &fs));
  CHECK(vkCreatePipelineLayout(
     dev,
     &(VkPipelineLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &(VkPushConstantRange){
           VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4}},
     NULL, &pl));
  const VkFormat fmt = FMT;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  CHECK(vkCreateGraphicsPipelines(
     dev, VK_NULL_HANDLE, 1,
     &(VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &(VkPipelineRenderingCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
           .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt},
        .stageCount = 2,
        .pStages = (VkPipelineShaderStageCreateInfo[]){
           {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
           {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs,
            .pName = "main"}},
        .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
        .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
           .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
           .viewportCount = 1, .scissorCount = 1},
        .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
           .lineWidth = 1},
        .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
           .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
        .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
           .attachmentCount = 1,
           .pAttachments = &(VkPipelineColorBlendAttachmentState){
              .colorWriteMask = 0xf}},
        .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
           .dynamicStateCount = 2, .pDynamicStates = dyn},
        .layout = pl},
     NULL, &pipe));

  CHECK(vkCreateCommandPool(
     dev,
     &(VkCommandPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT},
     NULL, &pool));
  CHECK(vkAllocateCommandBuffers(
     dev,
     &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = N},
     cbs));
  CHECK(vkCreateFence(
     dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
     NULL, &fence));

  /* Small (a mip-chain level), full screen at 3456x2234, and an atlas cell:
   * a 130x130 area of a 1040x1040 image, as Transport Fever 3 renders. */
  struct target small, full, atlas;
  make_target(&small, 64, 64);
  make_target(&full, 3456, 2234);
  make_target(&atlas, 1040, 1040);
  const struct target *all[] = {&small, &full, &atlas};
  begin(cbs[0]);
  for (unsigned t = 0; t < 3; t++)
    for (unsigned i = 0; i < 2; i++)
      barrier(cbs[0], all[t]->img[i], VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  CHECK(vkEndCommandBuffer(cbs[0]));
  CHECK(vkQueueSubmit(q, 1,
                      &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers = cbs},
                      fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  CHECK(vkResetFences(dev, 1, &fence));

  const VkAttachmentLoadOp L = VK_ATTACHMENT_LOAD_OP_LOAD,
                           C = VK_ATTACHMENT_LOAD_OP_CLEAR,
                           D = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  const VkRect2D whole = {{0, 0}, {0, 0}}, cell = {{390, 260}, {130, 130}};
  const struct bench benches[] = {
     {"64x64 clear 1 draw", &small, whole, C, 1, ONE_CB},
     {"64x64 load 1 draw", &small, whole, L, 1, ONE_CB},
     {"64x64 dontcare 1 draw", &small, whole, D, 1, ONE_CB},
     {"64x64 clear 0 draws", &small, whole, C, 0, ONE_CB},
     {"64x64 load 0 draws", &small, whole, L, 0, ONE_CB},
     {"64x64 load 2 draws", &small, whole, L, 2, ONE_CB},
     {"3456x2234 clear 1 draw", &full, whole, C, 1, ONE_CB},
     {"3456x2234 load 1 draw", &full, whole, L, 1, ONE_CB},
     {"3456x2234 dontcare 1 draw", &full, whole, D, 1, ONE_CB},
     {"3456x2234 load 0 draws", &full, whole, L, 0, ONE_CB},
     {"130x130 of 1040x1040 load 1 draw", &atlas, cell, L, 1, ONE_CB},
     {"130x130 of 1040x1040 clear 1 draw", &atlas, cell, C, 1, ONE_CB},
     {"64x64 clear 1 draw, N cmdbufs", &small, whole, C, 1, MANY_CB},
     {"64x64 clear 1 draw, N submits", &small, whole, C, 1, MANY_SUBMIT},
     {"64x64 load 1 draw, same image", &small, whole, L, 1, ONE_CB, true},
  };

  printf("%-36s %10s %10s %10s  (us per pass, %u reps, N=%u)\n", "case",
         "min", "median", "max", reps, N);
  for (unsigned b = 0; b < sizeof(benches) / sizeof(benches[0]); b++) {
    double us[64];
    for (unsigned r = 0; r < reps && r < 64; r++) {
      sleep_ms(30);
      double one = run(&benches[b], 1);
      sleep_ms(30);
      us[r] = (run(&benches[b], N) - one) * 1e3 / (N - 1);
    }
    qsort(us, reps, sizeof(double), cmp);
    printf("%-36s %10.1f %10.1f %10.1f\n", benches[b].name, us[0],
           us[reps / 2], us[reps - 1]);
    fflush(stdout);
  }

  /* Correctness: after the last two cases, both small images hold the clear
   * colour with the 32x32 draw in the corner. */
  VkBuffer buf;
  CHECK(vkCreateBuffer(dev,
                       &(VkBufferCreateInfo){
                          .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                          .size = 64 * 64 * 4 * 2,
                          .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT},
                       NULL, &buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buf, &mr);
  VkDeviceMemory bm;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex =
           memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)},
     NULL, &bm));
  CHECK(vkBindBufferMemory(dev, buf, bm, 0));
  CHECK(vkResetCommandBuffer(cbs[0], 0));
  begin(cbs[0]);
  for (unsigned i = 0; i < 2; i++) {
    barrier(cbs[0], small.img[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(
       cbs[0], small.img[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1,
       &(VkBufferImageCopy){.bufferOffset = i * 64 * 64 * 4,
                            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                                 0, 1},
                            .imageExtent = {64, 64, 1}});
  }
  CHECK(vkEndCommandBuffer(cbs[0]));
  CHECK(vkQueueSubmit(q, 1,
                      &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers = cbs},
                      fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  uint32_t *px;
  CHECK(vkMapMemory(dev, bm, 0, VK_WHOLE_SIZE, 0, (void **)&px));
  unsigned bad = 0;
  for (unsigned i = 0; i < 2 * 64 * 64; i++) {
    unsigned x = i % 64, y = (i / 64) % 64;
    uint32_t want = x < 32 && y < 32 ? colour : 0xffbf8040u;
    if (px[i] != want && bad++ < 8)
      fprintf(stderr, "image %u (%u, %u): %08x, expected %08x\n", i / 4096, x,
              y, px[i], want);
  }
  if (bad) {
    printf("FAIL: %u pixels wrong\n", bad);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
