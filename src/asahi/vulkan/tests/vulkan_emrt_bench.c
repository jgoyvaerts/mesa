/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Cost of spilled ("eMRT") multisample colour targets on G17P, in a pass
 * shaped like Transport Fever 3's G-buffer: R11G11B10F + A2R10G10B10 +
 * RGBA8 (12 bytes a sample) plus a D32 depth attachment, at 3456x2234. With
 * loadOp LOAD the tile keeps eight bytes a sample and RGBA8 spills; with
 * loadOp CLEAR all three stay in the tilebuffer.
 *
 * Each pass draws `layers` full-screen triangles, each one closer than the
 * last, so every layer passes the depth test:
 *   opaque   depth-tested, hidden-surface removal may drop covered layers
 *   discard  depth-tested, a quarter of each layer discarded
 *   blend    alpha blending on all targets, no depth writes
 *   a2c      depth-tested with alpha-to-coverage
 *
 *   aurora-emrt-bench [samples=2] [load|clear] [mode=blend] [layers=8]
 *                     [passes=20]
 *
 * One submission records `passes` passes over the same images. Take per-pass
 * GPU time from the asahi_g17_complete tracepoint; the printed wall-clock
 * time per pass is only a cross-check. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "vulkan_emrt_bench_shaders.h"

#define CHECK(x)                                                              \
   do {                                                                       \
      VkResult r_ = (x);                                                      \
      if (r_ != VK_SUCCESS) {                                                 \
         printf("FAIL: %s = %d (line %d)\n", #x, r_, __LINE__);               \
         exit(1);                                                             \
      }                                                                       \
   } while (0)

#define W 3456
#define H 2234
#define NUM_RT 3

static const VkFormat rt_formats[NUM_RT] = {
   VK_FORMAT_B10G11R11_UFLOAT_PACK32,
   VK_FORMAT_A2R10G10B10_UNORM_PACK32,
   VK_FORMAT_R8G8B8A8_UNORM,
};

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties memprops;

static VkImageView
make_view(VkFormat fmt, VkSampleCountFlagBits samples, VkImageUsageFlags usage,
          VkImageAspectFlags aspect)
{
   VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = {W, H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = samples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &ci, NULL, &img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, img, &req);
   uint32_t type = 0;
   while (!(req.memoryTypeBits & (1u << type)) ||
          !(memprops.memoryTypes[type].propertyFlags &
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      type++;
   VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL,
                              req.size, type};
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
   CHECK(vkBindImageMemory(dev, img, mem, 0));
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = fmt,
      .subresourceRange = {aspect, 0, 1, 0, 1},
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vi, NULL, &view));
   return view;
}

static VkShaderModule
make_module(const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = size, .pCode = code};
   VkShaderModule m;
   CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
   return m;
}

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int
main(int argc, char **argv)
{
   const unsigned samples = argc > 1 ? atoi(argv[1]) : 2;
   const bool load = !(argc > 2 && !strcmp(argv[2], "clear"));
   const char *mode = argc > 3 ? argv[3] : "blend";
   const unsigned layers = argc > 4 ? atoi(argv[4]) : 8;
   const unsigned passes = argc > 5 ? atoi(argv[5]) : 20;
   const bool blend = !strcmp(mode, "blend");
   const bool discard = !strcmp(mode, "discard");
   const bool a2c = !strcmp(mode, "a2c");
   if ((samples != 1 && samples != 2 && samples != 4) || !passes ||
       (!blend && !discard && !a2c && strcmp(mode, "opaque"))) {
      fprintf(stderr, "usage: %s [samples=1|2|4] [load|clear] "
                      "[opaque|discard|blend|a2c] [layers] [passes]\n",
              argv[0]);
      return 2;
   }
   printf("%ux%u@%u, loadOp %s, %s, %u layers, %u passes\n", W, H, samples,
          load ? "LOAD" : "CLEAR", mode, layers, passes);

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pdev;
   VkResult r = vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !n) {
      printf("FAIL: no physical device\n");
      return 1;
   }
   vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = 0, .queueCount = 1,
                                  .pQueuePriorities = &prio};
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &f13, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);

   VkImageView rt[NUM_RT];
   for (unsigned i = 0; i < NUM_RT; i++)
      rt[i] = make_view(rt_formats[i], samples,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
   VkImageView depth = make_view(VK_FORMAT_D32_SFLOAT, samples,
                                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                 VK_IMAGE_ASPECT_DEPTH_BIT);

   VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
   VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pcr};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));

   VkBool32 spec_discard = discard;
   VkSpecializationMapEntry spec_entry = {0, 0, sizeof(VkBool32)};
   VkSpecializationInfo spec = {1, &spec_entry, sizeof(spec_discard), &spec_discard};
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = make_module(emrt_bench_vert, sizeof(emrt_bench_vert)), .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = make_module(emrt_bench_frag, sizeof(emrt_bench_frag)), .pName = "main",
       .pSpecializationInfo = &spec},
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport vp = {0, 0, W, H, 0, 1};
   VkRect2D scissor = {{0, 0}, {W, H}};
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = samples, .alphaToCoverageEnable = a2c};
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = !blend,
      .depthCompareOp = VK_COMPARE_OP_LESS};
   VkPipelineColorBlendAttachmentState cba[NUM_RT];
   for (unsigned i = 0; i < NUM_RT; i++) {
      cba[i] = (VkPipelineColorBlendAttachmentState){
         .blendEnable = blend,
         .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
         .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
         .colorBlendOp = VK_BLEND_OP_ADD,
         .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
         .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
         .alphaBlendOp = VK_BLEND_OP_ADD,
         .colorWriteMask = 0xf};
   }
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = NUM_RT, .pAttachments = cba};
   VkPipelineRenderingCreateInfo rci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = NUM_RT, .pColorAttachmentFormats = rt_formats,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT};
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rci, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cbs, .layout = layout};
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipeline));

   VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                      .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      .commandBufferCount = 2};
   VkCommandBuffer cbs_[2];
   CHECK(vkAllocateCommandBuffers(dev, &cai, cbs_));

   VkRenderingAttachmentInfo att[NUM_RT];
   VkRenderingAttachmentInfo datt = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = depth,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .clearValue.depthStencil = {1.0f, 0},
   };
   VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                         .renderArea = {{0, 0}, {W, H}},
                         .layerCount = 1,
                         .colorAttachmentCount = NUM_RT,
                         .pColorAttachments = att,
                         .pDepthAttachment = &datt};
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
   };
   VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                          .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
   const float zstep = 1.0f / (layers + 1);

   /* Command buffer 0 clears the targets (and moves them to GENERAL);
    * command buffer 1 is the measured passes. */
   for (unsigned c = 0; c < 2; c++) {
      VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(cbs_[c], &cbi));
      for (unsigned p = 0; p < (c ? passes : 1); p++) {
         for (unsigned i = 0; i < NUM_RT; i++) {
            att[i] = (VkRenderingAttachmentInfo){
               .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = rt[i],
               .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .loadOp = c && load ? VK_ATTACHMENT_LOAD_OP_LOAD
                                   : VK_ATTACHMENT_LOAD_OP_CLEAR,
               .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
               .clearValue.color = {{0.25f, 0.5f, 0.75f, 1.0f}},
            };
         }
         vkCmdPipelineBarrier2(cbs_[c], &di);
         vkCmdBeginRendering(cbs_[c], &ri);
         if (c) {
            vkCmdBindPipeline(cbs_[c], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdPushConstants(cbs_[c], layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4,
                               &zstep);
            for (unsigned l = 0; l < layers; l++)
               vkCmdDraw(cbs_[c], 3, 1, 0, l);
         }
         vkCmdEndRendering(cbs_[c]);
      }
      CHECK(vkEndCommandBuffer(cbs_[c]));
   }

   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 2, .pCommandBuffers = cbs_};
   /* Warm-up (shader and allocation), then the timed run. */
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   const double t0 = now_ms();
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   printf("wall-clock: %.3f ms per pass\n", (now_ms() - t0) / (passes + 1));
   return 0;
}
