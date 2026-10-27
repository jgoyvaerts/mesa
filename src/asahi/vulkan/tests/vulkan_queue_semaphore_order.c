/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Submissions on one queue chained by binary semaphores, as games end a frame
 * with a submission that waits on the ones before it. The chain alternates
 * engines: a render pass whose fragment shader writes value + 1 into an
 * image, then a copy of that image back into the value buffer, each submitted
 * on its own and waiting on the semaphore the one before signalled. After N
 * links the value must be N. The driver orders such a wait in firmware after
 * the signalling submission's commands instead of passing the semaphore to
 * the kernel (hk_queue.c), so a missing or short ordering shows up as a lost
 * increment.
 *
 * It then times the chain against the same submissions without semaphores.
 * Waiting through the kernel drains the queue at every submission (on the
 * M4 Pro: ~110 us per chained submission; ~23 us with the wait ordered in
 * firmware, ~18 us without semaphores).
 *
 * Usage: vulkan-queue-semaphore-order   Exit status 0 = correct output.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "vulkan_queue_semaphore_order_shaders.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #call, r);       \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

#define SIZE 64
#define LINKS 100

static VkDevice dev;
static VkQueue queue;
static VkPhysicalDeviceMemoryProperties mem_props;

static VkDeviceMemory
alloc(VkMemoryRequirements req, VkMemoryPropertyFlags flags)
{
  uint32_t type = 0;
  while (!(req.memoryTypeBits & (1u << type)) ||
         (mem_props.memoryTypes[type].propertyFlags & flags) != flags)
    if (++type == mem_props.memoryTypeCount)
      exit(2);
  VkDeviceMemory mem;
  CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
     .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
     .memoryTypeIndex = type}, NULL, &mem));
  return mem;
}

static VkShaderModule
module(const uint32_t *code, size_t size)
{
  VkShaderModule m;
  CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){
     .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code},
     NULL, &m));
  return m;
}

static int
cmp_double(const void *a, const void *b)
{
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

int
main(void)
{
  VkInstance inst;
  CHECK(vkCreateInstance(&(VkInstanceCreateInfo){
     .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
     .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                              .apiVersion = VK_API_VERSION_1_3}}, NULL, &inst));
  uint32_t count = 1;
  VkPhysicalDevice pdev;
  vkEnumeratePhysicalDevices(inst, &count, &pdev);
  if (!count)
    return 2;
  vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
  CHECK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){
     .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
     .pNext = &(VkPhysicalDeviceVulkan13Features){
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .dynamicRendering = VK_TRUE},
     .queueCreateInfoCount = 1,
     .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
        .pQueuePriorities = &(float){1}}}, NULL, &dev));
  vkGetDeviceQueue(dev, 0, 0, &queue);

  /* The value buffer (read by the fragment shader, written by the copy) and
   * the image the render pass writes. */
  VkBuffer value;
  CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){
     .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 256,
     .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT}, NULL, &value));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, value, &req);
  VkDeviceMemory value_mem = alloc(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  CHECK(vkBindBufferMemory(dev, value, value_mem, 0));
  volatile uint32_t *map;
  CHECK(vkMapMemory(dev, value_mem, 0, 256, 0, (void **)&map));

  VkImage img;
  CHECK(vkCreateImage(dev, &(VkImageCreateInfo){
     .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
     .format = VK_FORMAT_R32_UINT, .extent = {SIZE, SIZE, 1}, .mipLevels = 1, .arrayLayers = 1,
     .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
     .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
     NULL, &img));
  vkGetImageMemoryRequirements(dev, img, &req);
  CHECK(vkBindImageMemory(dev, img, alloc(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
  VkImageView view;
  CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){
     .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
     .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_UINT,
     .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &view));

  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
     .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
     .pBindings = &(VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                  VK_SHADER_STAGE_FRAGMENT_BIT}}, NULL, &dsl));
  VkPipelineLayout layout;
  CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
     .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
     .pSetLayouts = &dsl}, NULL, &layout));
  VkDescriptorPool dpool;
  CHECK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
     .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1,
     .pPoolSizes = &(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}}, NULL, &dpool));
  VkDescriptorSet set;
  CHECK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
     .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool,
     .descriptorSetCount = 1, .pSetLayouts = &dsl}, &set));
  vkUpdateDescriptorSets(dev, 1, &(VkWriteDescriptorSet){
     .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .descriptorCount = 1,
     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     .pBufferInfo = &(VkDescriptorBufferInfo){value, 0, VK_WHOLE_SIZE}}, 0, NULL);

  const VkFormat fmt = VK_FORMAT_R32_UINT;
  VkPipeline pipe;
  CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
     .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
     .pNext = &(VkPipelineRenderingCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &fmt},
     .stageCount = 2,
     .pStages = (VkPipelineShaderStageCreateInfo[]){
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .pName = "main",
         .module = module(queue_semaphore_order_vert, sizeof(queue_semaphore_order_vert))},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .pName = "main",
         .module = module(queue_semaphore_order_frag, sizeof(queue_semaphore_order_frag))}},
     .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
     .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
     .pViewportState = &(VkPipelineViewportStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1,
        .pViewports = &(VkViewport){0, 0, SIZE, SIZE, 0, 1}, .scissorCount = 1,
        .pScissors = &(VkRect2D){{0, 0}, {SIZE, SIZE}}},
     .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .lineWidth = 1},
     .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
     .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
        .pAttachments = &(VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf}},
     .layout = layout}, NULL, &pipe));

  VkCommandPool pool;
  CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){
     .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}, NULL, &pool));
  /* cb[0]: image to GENERAL; cb[1]: the render pass; cb[2]: the copy back. */
  VkCommandBuffer cb[3];
  CHECK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
     .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
     .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 3}, cb));
  for (unsigned i = 0; i < 3; ++i)
    CHECK(vkBeginCommandBuffer(cb[i], &(VkCommandBufferBeginInfo){
       .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
       .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT}));
  vkCmdPipelineBarrier(cb[0], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1,
                       &(VkImageMemoryBarrier){
                          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                          .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                          .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = img,
                          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
  vkCmdBindDescriptorSets(cb[1], VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
  vkCmdBeginRendering(cb[1], &(VkRenderingInfo){
     .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {SIZE, SIZE}},
     .layerCount = 1, .colorAttachmentCount = 1,
     .pColorAttachments = &(VkRenderingAttachmentInfo){
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
  vkCmdBindPipeline(cb[1], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  vkCmdDraw(cb[1], 3, 1, 0, 0);
  vkCmdEndRendering(cb[1]);
  vkCmdCopyImageToBuffer(cb[2], img, VK_IMAGE_LAYOUT_GENERAL, value, 1, &(VkBufferImageCopy){
     .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {1, 1, 1}});
  for (unsigned i = 0; i < 3; ++i)
    CHECK(vkEndCommandBuffer(cb[i]));

  VkSemaphore sem[2];
  for (unsigned i = 0; i < 2; ++i)
    CHECK(vkCreateSemaphore(dev, &(VkSemaphoreCreateInfo){
       .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}, NULL, &sem[i]));
  VkFence fence;
  CHECK(vkCreateFence(dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
                      NULL, &fence));
  CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
     .commandBufferCount = 1, .pCommandBuffers = &cb[0]}, fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  CHECK(vkResetFences(dev, 1, &fence));

  /* Submission k is a render pass for even k, a copy for odd k; with chained,
   * it waits on what submission k-1 signalled. */
  const VkPipelineStageFlags stage[2] = {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT};
  bool ok = true;
  double t[2][9];
  for (unsigned chained = 0; chained < 2; ++chained) {
    for (unsigned round = 0; round < 9; ++round) {
      map[0] = 0;
      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      for (unsigned k = 0; k < 2 * LINKS; ++k) {
        CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){
           .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
           .waitSemaphoreCount = chained && k > 0, .pWaitSemaphores = &sem[(k + 1) & 1],
           .pWaitDstStageMask = &stage[k & 1], .commandBufferCount = 1,
           .pCommandBuffers = &cb[1 + (k & 1)],
           .signalSemaphoreCount = chained && k + 1 < 2 * LINKS,
           .pSignalSemaphores = &sem[k & 1]}, k + 1 == 2 * LINKS ? fence : VK_NULL_HANDLE));
      }
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
      clock_gettime(CLOCK_MONOTONIC, &t1);
      CHECK(vkResetFences(dev, 1, &fence));
      t[chained][round] = ((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3) /
                          (2 * LINKS);
      if (chained && map[0] != LINKS) {
        fprintf(stderr, "round %u: value is %u after %u chained links\n", round, map[0], LINKS);
        ok = false;
      }
    }
    qsort(t[chained], 9, sizeof(double), cmp_double);
  }
  printf("%s; per submission: unchained %.1f us, chained by a binary semaphore %.1f us\n",
         ok ? "chained increments correct" : "FAIL: chained increments lost", t[0][4], t[1][4]);
  return ok ? 0 : 1;
}
