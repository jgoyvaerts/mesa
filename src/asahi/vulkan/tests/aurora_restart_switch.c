/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Primitive restart with the restart index changing between draws inside one
 * render pass (VK_EXT_primitive_restart_index on one bound u32 index buffer).
 * G17P emits the restart State Update only when the stream's restart index
 * may have changed (after a bind tail or an indirect draw, which writes its
 * own), so draws 2 and 5 below restart on state an earlier record left.
 * Each draw is a strip quad, restart, quad:
 *
 *   1. restart index K1
 *   2. K1, no state change (no restart record of its own)
 *   3. K2
 *   4. K2, indexed indirect
 *   5. K2, direct after the indirect packet
 *
 * Vertex v is at grid point (v & 0xff, v >> 8) of a 256x256 image, so a
 * restart index taken as a vertex draws triangles out to K1's (240, 10) or
 * K2's (10, 240).
 *
 * Usage: aurora-restart-switch   Exit status 0 = correct output.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "aurora_restart_switch_shaders.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

#define S 256
#define V(x, y) ((uint32_t)(y) << 8 | (uint32_t)(x))
/* Strip order of the quad with corners (x0, y0) and (x1, y1). */
#define QUAD(x0, y0, x1, y1) V(x0, y0), V(x1, y0), V(x0, y1), V(x1, y1)

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
  VkDeviceMemory mem;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex =
           memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)},
     NULL, &mem));
  CHECK(vkBindBufferMemory(dev, buf, mem, 0));
  CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, map));
  return buf;
}

/* Pixel (x, y) of the image for grid point (gx, gy). */
static unsigned px(unsigned g) { return g * S / 255; }

int main(void) {
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
  VkPhysicalDevicePrimitiveRestartIndexFeaturesEXT pri = {
     .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_RESTART_INDEX_FEATURES_EXT};
  vkGetPhysicalDeviceFeatures2(pdev, &(VkPhysicalDeviceFeatures2){
                                        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                        .pNext = &pri});
  if (!pri.primitiveRestartIndex) {
    printf("SKIP: no primitiveRestartIndex\n");
    return 77;
  }
  VkPhysicalDeviceVulkan13Features f13 = {
     .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
     .pNext = &pri, .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE};
  CHECK(vkCreateDevice(
     pdev,
     &(VkDeviceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f13,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames =
           (const char *[]){VK_EXT_PRIMITIVE_RESTART_INDEX_EXTENSION_NAME},
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
           .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
           .queueCount = 1, .pQueuePriorities = &prio}},
     NULL, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, 0, 0, &q);

  const uint32_t k1 = V(240, 10), k2 = V(10, 240);
  /* One strip per draw, in draw order. */
  const uint32_t idx[5][9] = {
     {QUAD(10, 10, 30, 30), k1, QUAD(10, 60, 30, 80)},
     {QUAD(60, 100, 80, 120), k1, QUAD(60, 150, 80, 170)},
     {QUAD(110, 10, 130, 30), k2, QUAD(110, 60, 130, 80)},
     {QUAD(160, 100, 180, 120), k2, QUAD(160, 150, 180, 170)},
     {QUAD(200, 150, 220, 170), k2, QUAD(200, 200, 220, 220)},
  };
  void *imap, *rmap, *dmap;
  VkBuffer ib = host_buffer(sizeof(idx), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &imap);
  memcpy(imap, idx, sizeof(idx));
  VkBuffer indirect = host_buffer(
     sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
     &dmap);
  *(VkDrawIndexedIndirectCommand *)dmap =
     (VkDrawIndexedIndirectCommand){9, 1, 27, 0, 0};
  VkBuffer rb = host_buffer(S * S * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rmap);

  const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
  VkImage img;
  CHECK(vkCreateImage(
     dev,
     &(VkImageCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = {S, S, 1},
        .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
     NULL, &img));
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, img, &mr);
  VkDeviceMemory im;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex =
           memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
     NULL, &im));
  CHECK(vkBindImageMemory(dev, img, im, 0));
  VkImageView view;
  CHECK(vkCreateImageView(
     dev,
     &(VkImageViewCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
     NULL, &view));

  VkShaderModule vs, fs;
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(restart_switch_vert), .pCode = restart_switch_vert},
     NULL, &vs));
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(restart_switch_frag), .pCode = restart_switch_frag},
     NULL, &fs));
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(
     dev,
     &(VkPipelineLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO},
     NULL, &pl));
  VkPipeline pipe;
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
           .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
           .primitiveRestartEnable = VK_TRUE},
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
           .viewportCount = 1, .pViewports = &(VkViewport){0, 0, S, S, 0, 1},
           .scissorCount = 1, .pScissors = &(VkRect2D){{0, 0}, {S, S}}},
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
        .layout = pl},
     NULL, &pipe));

  VkCommandPool pool;
  CHECK(vkCreateCommandPool(
     dev,
     &(VkCommandPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO},
     NULL, &pool));
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(
     dev,
     &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1},
     &cb));
  CHECK(vkBeginCommandBuffer(
     cb, &(VkCommandBufferBeginInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
  VkImageMemoryBarrier2 bar = {
     .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
     .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
     .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
     .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = img,
     .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
                               .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                               .imageMemoryBarrierCount = 1,
                               .pImageMemoryBarriers = &bar});
  vkCmdBeginRendering(
     cb, &(VkRenderingInfo){
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {S, S}}, .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &(VkRenderingAttachmentInfo){
               .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
               .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  vkCmdBindIndexBuffer(cb, ib, 0, VK_INDEX_TYPE_UINT32);
  PFN_vkCmdSetPrimitiveRestartIndexEXT set_restart_index =
     (PFN_vkCmdSetPrimitiveRestartIndexEXT)vkGetDeviceProcAddr(
        dev, "vkCmdSetPrimitiveRestartIndexEXT");
  set_restart_index(cb, k1);
  vkCmdDrawIndexed(cb, 9, 1, 0, 0, 0);
  vkCmdDrawIndexed(cb, 9, 1, 9, 0, 0);
  set_restart_index(cb, k2);
  vkCmdDrawIndexed(cb, 9, 1, 18, 0, 0);
  vkCmdDrawIndexedIndirect(cb, indirect, 0, 1, 0);
  vkCmdDrawIndexed(cb, 9, 1, 36, 0, 0);
  vkCmdEndRendering(cb);
  bar.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  bar.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  bar.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
  bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
                               .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                               .imageMemoryBarrierCount = 1,
                               .pImageMemoryBarriers = &bar});
  vkCmdCopyImageToBuffer(
     cb, img, VK_IMAGE_LAYOUT_GENERAL, rb, 1,
     &(VkBufferImageCopy){0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          {0, 0, 0}, {S, S, 1}});
  CHECK(vkEndCommandBuffer(cb));
  VkFence fence;
  CHECK(vkCreateFence(
     dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
     NULL, &fence));
  CHECK(vkQueueSubmit(q, 1,
                      &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers = &cb},
                      fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

  /* Grid points that must be covered (quad centres) or empty (between the
   * quads of a strip, and centroids of the triangles out to the other
   * restart index's vertex). */
  static const struct {
    unsigned x, y;
    bool covered;
    const char *what;
  } probes[] = {
     {20, 20, true, "draw 1 first quad"},
     {20, 70, true, "draw 1 second quad"},
     {20, 45, false, "draw 1 restart"},
     {93, 23, false, "draw 1 restart (triangle to K1)"},
     {70, 110, true, "draw 2 first quad"},
     {70, 160, true, "draw 2 second quad"},
     {70, 135, false, "draw 2 restart"},
     {127, 93, false, "draw 2 restart (triangle to K1)"},
     {120, 20, true, "draw 3 first quad"},
     {120, 70, true, "draw 3 second quad"},
     {120, 45, false, "draw 3 restart"},
     {83, 110, false, "draw 3 restart (triangle to K2)"},
     {170, 110, true, "draw 4 indirect first quad"},
     {170, 160, true, "draw 4 indirect second quad"},
     {170, 135, false, "draw 4 indirect restart"},
     {210, 160, true, "draw 5 first quad"},
     {210, 210, true, "draw 5 second quad"},
     {210, 185, false, "draw 5 restart"},
     {143, 193, false, "draw 5 restart (triangle to K2)"},
  };
  const uint32_t *pix = rmap;
  unsigned bad = 0;
  for (unsigned i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
    const uint32_t v = pix[px(probes[i].y) * S + px(probes[i].x)];
    if ((v == 0xffffffffu) != probes[i].covered) {
      fprintf(stderr, "%s: pixel (%u, %u) = %08x\n", probes[i].what,
              px(probes[i].x), px(probes[i].y), v);
      bad++;
    }
  }
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad ? 1 : 0;
}
