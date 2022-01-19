/*
* Vulkan Example - Compute shader culling and LOD using indirect rendering
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"
#include "frustum.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define INSTANCE_BUFFER_BIND_ID 1
#define ENABLE_VALIDATION false

// Total number of objects (^3) in the scene
#if defined(__ANDROID__)
#define OBJECT_COUNT 32
#else
#define OBJECT_COUNT 100
#endif

#define MAX_LOD_LEVEL 5

constexpr uint32_t ceil2(uint32_t x)
{
    --x;
    x >>= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

struct alignas(16) ImageWidthPushConstant
{
    glm::vec2 imageSize;
};

class VulkanExample : public VulkanExampleBase
{
public:
    PFN_vkCmdDrawIndexedIndirectCountKHR vkCmdDrawIndexedIndirectCountKHR;

public:
    bool fixedFrustum = false;

    // The model contains multiple versions of a single object with different levels of detail
    vkglTF::Model lodModel;

    // Per-instance data block
    struct InstanceData {
        glm::vec3 pos;
        float scale;
    };

    // Contains the instanced data
    vks::Buffer instanceBuffer;
    // Contains the indirect drawing commands
    vks::Buffer indirectCommandsBuffer;
    vks::Buffer indirectDrawCountBuffer;

    // Indirect draw statistics (updated via compute)
    struct {
        uint32_t drawCount;						// Total number of indirect draw counts to be issued
        uint32_t occluded;
        uint32_t primitiveCount;
        uint32_t lodCount[MAX_LOD_LEVEL + 1];	// Statistics for number of draws per LOD level (written by compute shader)
    } indirectStats;

    // Store the indirect draw commands containing index offsets and instance count per object
    std::vector<VkDrawIndexedIndirectCommand> indirectCommands;

    struct {
        glm::mat4 projection;
        glm::mat4 modelview;
        glm::vec4 cameraPos;
        glm::vec4 frustumPlanes[6];
    } uboScene;

    struct {
        vks::Buffer scene;
    } uniformData;

    struct {
        VkPipeline plants;
    } pipelines;

    VkPipelineLayout pipelineLayout;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;



    // Resources for the compute part of the example
    struct {
        vks::Buffer lodLevelsBuffers;				// Contains index start and counts for the different lod levels
        VkQueue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
        VkCommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
        VkCommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
        VkFence fence;								// Synchronization fence to avoid rewriting compute CB if still in use
        VkSemaphore semaphore;						// Used as a wait semaphore for graphics submission
        VkDescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
        VkDescriptorSet descriptorSet;				// Compute shader bindings
        VkPipelineLayout pipelineLayout;			// Layout of the compute pipeline
        VkPipeline pipeline;						// Compute pipeline for updating particle positions

        VkDescriptorSetLayout depthPyramidLevelDescriptorSetLayout;
        std::vector<VkDescriptorSet> depthPyramidLevelDescriptorSets;				// Compute shader bindings
        VkPipelineLayout depthPyramidPipelineLayout;			// Layout of the compute pipeline
        VkPipeline depthPyramidPipeline;						// Compute pipeline for updating particle positions
        VkSampler depthSampler;
    } compute;

    // View frustum for culling invisible objects
    vks::Frustum frustum;

    uint32_t objectCount = 0;
    uint32_t depthLevelCount = 0;

    VkDescriptorPool depthPyramidDescriptorPool;
    VkImage depthPyramidImage;
    VkDeviceMemory depthPyramidImageMemory;
    std::vector<VkImageView> depthPyramidLevelViews;
    VkImageView depthPyramidImageView;

    VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
    {
        apiVersion = VK_API_VERSION_1_1;
        title = "Vulkan Example - Compute cull and lod";
        camera.type = Camera::CameraType::firstperson;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);

        camera.setTranslation(glm::vec3(0.5f, 0.0f, 0.0f));
        camera.movementSpeed = 5.0f;
        memset(&indirectStats, 0, sizeof(indirectStats));
        enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        enabledInstanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
    }

    ~VulkanExample()
    {
        vkDestroyPipeline(device, pipelines.plants, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        instanceBuffer.destroy();
        indirectCommandsBuffer.destroy();
        uniformData.scene.destroy();
        indirectDrawCountBuffer.destroy();
        compute.lodLevelsBuffers.destroy();
        vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
        vkDestroyPipeline(device, compute.pipeline, nullptr);
        vkDestroyFence(device, compute.fence, nullptr);
        vkDestroyCommandPool(device, compute.commandPool, nullptr);
        vkDestroySemaphore(device, compute.semaphore, nullptr);
        vkDestroyPipeline(device, compute.depthPyramidPipeline, nullptr);
        vkDestroyDescriptorPool(device, depthPyramidDescriptorPool, nullptr);
        vkDestroyPipelineLayout(device, compute.depthPyramidPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, compute.depthPyramidLevelDescriptorSetLayout, nullptr);
        vkDestroySampler(device, compute.depthSampler, nullptr);
        for (auto imageView : depthPyramidLevelViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroyImageView(device, depthPyramidImageView, nullptr);
        vkDestroyImage(device, depthPyramidImage, nullptr);
        vkFreeMemory(device, depthPyramidImageMemory, nullptr);
    }

    virtual void getEnabledFeatures()
    {
        // Enable multi draw indirect if supported
        if (deviceFeatures.multiDrawIndirect) {
            enabledFeatures.multiDrawIndirect = VK_TRUE;
        }
    }


    void setupRenderPass() final
    {
        std::array<VkAttachmentDescription, 2> attachments = {};
        // Color attachment
        attachments[0].format = swapChain.colorFormat;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        // Depth attachment
        attachments[1].format = depthFormat;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorReference = {};
        colorReference.attachment = 0;
        colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthReference = {};
        depthReference.attachment = 1;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;
        subpassDescription.pDepthStencilAttachment = &depthReference;
        subpassDescription.inputAttachmentCount = 0;
        subpassDescription.pInputAttachments = nullptr;
        subpassDescription.preserveAttachmentCount = 0;
        subpassDescription.pPreserveAttachments = nullptr;
        subpassDescription.pResolveAttachments = nullptr;

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask = 0;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDescription;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
    }

    void setupDepthStencil() final
    {
        VkImageCreateInfo imageCI{};
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = depthFormat;
        imageCI.extent = { width, height, 1 };
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &depthStencil.image));
        VkMemoryRequirements memReqs{};
        vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);

        VkMemoryAllocateInfo memAllloc{};
        memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAllloc.allocationSize = memReqs.size;
        memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &depthStencil.mem));
        VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0));

        VkImageViewCreateInfo imageViewCI{};
        imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = depthStencil.image;
        imageViewCI.format = depthFormat;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        /*if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }*/
        VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &depthStencil.view));
    }

    void createDepthPyramidResources()
    {
        depthLevelCount = uint32_t(std::log2(std::max(width, height)));
        VkImageCreateInfo imageCI{};
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = VK_FORMAT_R32_SFLOAT;
        imageCI.extent = { ceil2(width), ceil2(height), 1 };
        imageCI.mipLevels = depthLevelCount;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &depthPyramidImage));
        VkMemoryRequirements memReqs{};
        vkGetImageMemoryRequirements(device, depthPyramidImage, &memReqs);

        VkMemoryAllocateInfo memAllloc{};
        memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAllloc.allocationSize = memReqs.size;
        memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &depthPyramidImageMemory));
        VK_CHECK_RESULT(vkBindImageMemory(device, depthPyramidImage, depthPyramidImageMemory, 0));


        depthPyramidLevelViews.clear();
        depthPyramidLevelViews.resize(depthLevelCount);
        for (uint32_t level = 0; level < depthLevelCount; ++level)
        {
            VkImageViewCreateInfo imageViewCI{};
            imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewCI.image = depthPyramidImage;
            imageViewCI.format = VK_FORMAT_R32_SFLOAT;
            imageViewCI.subresourceRange.baseMipLevel = level;
            imageViewCI.subresourceRange.levelCount = 1;
            imageViewCI.subresourceRange.baseArrayLayer = 0;
            imageViewCI.subresourceRange.layerCount = 1;
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
            VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &depthPyramidLevelViews[level]));
        }

        VkImageViewCreateInfo depthPyramidImageViewCI{};
        depthPyramidImageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthPyramidImageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthPyramidImageViewCI.image = depthPyramidImage;
        depthPyramidImageViewCI.format = VK_FORMAT_R32_SFLOAT;
        depthPyramidImageViewCI.subresourceRange.baseMipLevel = 0;
        depthPyramidImageViewCI.subresourceRange.levelCount = depthLevelCount;
        depthPyramidImageViewCI.subresourceRange.baseArrayLayer = 0;
        depthPyramidImageViewCI.subresourceRange.layerCount = 1;
        depthPyramidImageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        VK_CHECK_RESULT(vkCreateImageView(device, &depthPyramidImageViewCI, nullptr, &depthPyramidImageView));

        VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.minLod = 0.f;
        samplerInfo.maxLod = 16.f;
//        samplerInfo.mipLodBias = -0.5f;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VkSamplerReductionModeCreateInfo createInfoReduction = { VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO };
        createInfoReduction.reductionMode = VK_SAMPLER_REDUCTION_MODE_MAX;
        samplerInfo.pNext = &createInfoReduction;

        VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &compute.depthSampler));

        vkResetDescriptorPool(device, depthPyramidDescriptorPool, 0);

    }

    void buildCommandBuffers()
    {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VkClearValue clearValues[2];
        clearValues[0].color = { { 0.18f, 0.27f, 0.5f, 0.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.extent.width = width;
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
        {
            // Set target frame buffer
            renderPassBeginInfo.framebuffer = frameBuffers[i];

            VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

            VkImageMemoryBarrier depthImageAcquireBarrier = vks::initializers::imageMemoryBarrier();
            depthImageAcquireBarrier.image = depthStencil.image;
            depthImageAcquireBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthImageAcquireBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthImageAcquireBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
                depthImageAcquireBarrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            depthImageAcquireBarrier.subresourceRange.baseArrayLayer = 0;
            depthImageAcquireBarrier.subresourceRange.layerCount = 1;
            depthImageAcquireBarrier.subresourceRange.baseMipLevel = 0;
            depthImageAcquireBarrier.subresourceRange.levelCount = 1;
            depthImageAcquireBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
            depthImageAcquireBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
            depthImageAcquireBarrier.srcAccessMask = 0;
            depthImageAcquireBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            VkBufferMemoryBarrier indirectCommandsBuffersBarriers[2] = { vks::initializers::bufferMemoryBarrier(), vks::initializers::bufferMemoryBarrier() };
            indirectCommandsBuffersBarriers[0].srcAccessMask = 0;
            indirectCommandsBuffersBarriers[0].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            indirectCommandsBuffersBarriers[0].buffer = indirectCommandsBuffer.buffer;
            indirectCommandsBuffersBarriers[0].size = indirectCommandsBuffer.descriptor.range;
            indirectCommandsBuffersBarriers[0].srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
            indirectCommandsBuffersBarriers[0].dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;

            indirectCommandsBuffersBarriers[1].srcAccessMask = 0;
            indirectCommandsBuffersBarriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            indirectCommandsBuffersBarriers[1].buffer = indirectDrawCountBuffer.buffer;
            indirectCommandsBuffersBarriers[1].size = indirectDrawCountBuffer.descriptor.range;
            indirectCommandsBuffersBarriers[1].srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
            indirectCommandsBuffersBarriers[1].dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;

            vkCmdPipelineBarrier(
                drawCmdBuffers[i],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                VK_FLAGS_NONE,
                0, nullptr,
                std::size(indirectCommandsBuffersBarriers), indirectCommandsBuffersBarriers,
                1, &depthImageAcquireBarrier);

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

            // Mesh containing the LODs
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.plants);
//            vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &lodModel.vertices.buffer, offsets);
            vkCmdBindVertexBuffers(drawCmdBuffers[i], INSTANCE_BUFFER_BIND_ID, 1, &instanceBuffer.buffer, offsets);

            vkCmdBindIndexBuffer(drawCmdBuffers[i], lodModel.indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexedIndirectCountKHR(drawCmdBuffers[i], indirectCommandsBuffer.buffer, 0, indirectDrawCountBuffer.buffer, 0, indirectCommands.size(), sizeof(VkDrawIndexedIndirectCommand));

            drawUI(drawCmdBuffers[i]);

            vkCmdEndRenderPass(drawCmdBuffers[i]);

            VkImageMemoryBarrier depthImageReleaseBarrier = vks::initializers::imageMemoryBarrier();
            depthImageAcquireBarrier.image = depthStencil.image;
            depthImageAcquireBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthImageAcquireBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthImageAcquireBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
                depthImageAcquireBarrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            depthImageAcquireBarrier.subresourceRange.baseArrayLayer = 0;
            depthImageAcquireBarrier.subresourceRange.layerCount = 1;
            depthImageAcquireBarrier.subresourceRange.baseMipLevel = 0;
            depthImageAcquireBarrier.subresourceRange.levelCount = 1;
            depthImageAcquireBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
            depthImageAcquireBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
            depthImageAcquireBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depthImageAcquireBarrier.dstAccessMask = 0;

            VkBufferMemoryBarrier indirectBufferReleaseBarrier = vks::initializers::bufferMemoryBarrier();
            indirectBufferReleaseBarrier.buffer = indirectCommandsBuffer.buffer;
            indirectBufferReleaseBarrier.size = indirectCommandsBuffer.descriptor.range;
            indirectBufferReleaseBarrier.srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            indirectBufferReleaseBarrier.dstAccessMask = 0;
            indirectBufferReleaseBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
            indirectBufferReleaseBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;

            VkBufferMemoryBarrier countBufferReleaseBarrier = vks::initializers::bufferMemoryBarrier();
            countBufferReleaseBarrier.buffer = indirectDrawCountBuffer.buffer;
            countBufferReleaseBarrier.size = indirectDrawCountBuffer.descriptor.range;
            countBufferReleaseBarrier.srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            countBufferReleaseBarrier.dstAccessMask = 0;
            countBufferReleaseBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
            countBufferReleaseBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;

            const VkImageMemoryBarrier imageBarriers[] = { depthImageAcquireBarrier };
            const VkBufferMemoryBarrier bufferBarriers[] = { indirectBufferReleaseBarrier, countBufferReleaseBarrier };

            vkCmdPipelineBarrier(
                drawCmdBuffers[i],
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_FLAGS_NONE,
                0, nullptr,
                std::size(bufferBarriers), bufferBarriers,
                1, imageBarriers);

            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }

    void loadAssets()
    {
        const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
        lodModel.loadFromFile(getAssetPath() + "models/suzanne_lods.gltf", vulkanDevice, queue, glTFLoadingFlags);
    }

    void buildComputeCommandBuffer()
    {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));
        VkImageMemoryBarrier depthPyramidLayoutBarrier = vks::initializers::imageMemoryBarrier();
        depthPyramidLayoutBarrier.image = depthPyramidImage;
        depthPyramidLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthPyramidLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthPyramidLayoutBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        depthPyramidLayoutBarrier.subresourceRange.baseArrayLayer = 0;
        depthPyramidLayoutBarrier.subresourceRange.layerCount = 1;
        depthPyramidLayoutBarrier.subresourceRange.baseMipLevel = 0;
        depthPyramidLayoutBarrier.subresourceRange.levelCount = depthLevelCount;
        depthPyramidLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthPyramidLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthPyramidLayoutBarrier.srcAccessMask = 0;
        depthPyramidLayoutBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

        VkImageMemoryBarrier depthImageAcquireBarrier = vks::initializers::imageMemoryBarrier();
        depthImageAcquireBarrier.image = depthStencil.image;
        depthImageAcquireBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthImageAcquireBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthImageAcquireBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            depthImageAcquireBarrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        depthImageAcquireBarrier.subresourceRange.baseArrayLayer = 0;
        depthImageAcquireBarrier.subresourceRange.layerCount = 1;
        depthImageAcquireBarrier.subresourceRange.baseMipLevel = 0;
        depthImageAcquireBarrier.subresourceRange.levelCount = 1;
        depthImageAcquireBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
        depthImageAcquireBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
        depthImageAcquireBarrier.srcAccessMask = 0;
        depthImageAcquireBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        const VkImageMemoryBarrier imageBarriers[] = { depthPyramidLayoutBarrier, depthImageAcquireBarrier };
        vkCmdPipelineBarrier(
            compute.commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_FLAGS_NONE,
            0, nullptr,
            0, nullptr,
            2, imageBarriers);
        uint32_t levelWidth = ceil2(width);
        uint32_t levelHeight = ceil2(height);

        auto workGroupSize = [](uint32_t elements) {
            const uint32_t localSize = 32;
            return (elements + localSize - 1) / localSize;
        };
        
        vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.depthPyramidPipeline);
        vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.depthPyramidPipelineLayout, 0, 1, &compute.depthPyramidLevelDescriptorSets[0], 0, 0);
        ImageWidthPushConstant push{ glm::vec2(float(levelWidth), float(levelHeight)) };
        vkCmdPushConstants(compute.commandBuffer, compute.depthPyramidPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ImageWidthPushConstant), &push);
        
        vkCmdDispatch(compute.commandBuffer, workGroupSize(levelWidth), workGroupSize(levelHeight), 1);
        for (uint32_t level = 1; level < depthLevelCount; ++level)
        {
            levelWidth /= 2;
            levelHeight /= 2;
            push.imageSize = glm::vec2(float(levelWidth), float(levelHeight));
            VkMemoryBarrier depthLevelBarrier = vks::initializers::memoryBarrier();
            depthLevelBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            depthLevelBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                compute.commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_FLAGS_NONE,
                1, &depthLevelBarrier,
                0, nullptr,
                0, nullptr);
            vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.depthPyramidPipelineLayout, 0, 1, &compute.depthPyramidLevelDescriptorSets[level], 0, 0);
            vkCmdPushConstants(compute.commandBuffer, compute.depthPyramidPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ImageWidthPushConstant), &push);
            vkCmdDispatch(compute.commandBuffer, workGroupSize(levelWidth), workGroupSize(levelHeight), 1);
        }

        VkMemoryBarrier depthPyramidCompleteBarrier = vks::initializers::memoryBarrier();
        depthPyramidCompleteBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        depthPyramidCompleteBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            compute.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_FLAGS_NONE,
            1, &depthPyramidCompleteBarrier,
            0, nullptr,
            0, nullptr);

        VkImageMemoryBarrier depthImageReleaseBarrier = vks::initializers::imageMemoryBarrier();
        depthImageReleaseBarrier.image = depthStencil.image;
        depthImageReleaseBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthImageReleaseBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthImageReleaseBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            depthImageReleaseBarrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        depthImageReleaseBarrier.subresourceRange.baseArrayLayer = 0;
        depthImageReleaseBarrier.subresourceRange.layerCount = 1;
        depthImageReleaseBarrier.subresourceRange.baseMipLevel = 0;
        depthImageReleaseBarrier.subresourceRange.levelCount = 1;
        depthImageReleaseBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
        depthImageReleaseBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
        depthImageReleaseBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        depthImageReleaseBarrier.dstAccessMask = 0;
        /*vkCmdPipelineBarrier(
            compute.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_FLAGS_NONE,
            0, nullptr,
            0, nullptr,
            1, &depthImageReleaseBarrier);*/


        // Add memory barrier to ensure that the indirect commands have been consumed before the compute shader updates them
        VkBufferMemoryBarrier indirectBufferAcquireBarrier = vks::initializers::bufferMemoryBarrier();
        indirectBufferAcquireBarrier.buffer = indirectCommandsBuffer.buffer;
        indirectBufferAcquireBarrier.size = indirectCommandsBuffer.descriptor.range;
        indirectBufferAcquireBarrier.srcAccessMask = 0;
        indirectBufferAcquireBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        indirectBufferAcquireBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
        indirectBufferAcquireBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;

        VkBufferMemoryBarrier countBufferAcquireBarrier = vks::initializers::bufferMemoryBarrier();
        countBufferAcquireBarrier.buffer = indirectDrawCountBuffer.buffer;
        countBufferAcquireBarrier.size = indirectDrawCountBuffer.descriptor.range;
        countBufferAcquireBarrier.srcAccessMask = 0;
        countBufferAcquireBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        countBufferAcquireBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
        countBufferAcquireBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;

        VkBufferMemoryBarrier indirectCommandsBuffersBarriers[] = { indirectBufferAcquireBarrier, countBufferAcquireBarrier };

        vkCmdPipelineBarrier(
            compute.commandBuffer,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_FLAGS_NONE,
            0, nullptr,
            2, indirectCommandsBuffersBarriers,
            0, nullptr);

        vkCmdFillBuffer(compute.commandBuffer, indirectDrawCountBuffer.buffer, 0, indirectDrawCountBuffer.descriptor.range, 0u);

        VkBufferMemoryBarrier countBufferClearBarrier = vks::initializers::bufferMemoryBarrier();
        countBufferClearBarrier.buffer = indirectDrawCountBuffer.buffer;
        countBufferClearBarrier.size = indirectDrawCountBuffer.descriptor.range;
        countBufferClearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        countBufferClearBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        countBufferClearBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        countBufferClearBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(
            compute.commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_FLAGS_NONE,
            0, nullptr,
            1, &countBufferClearBarrier,
            0, nullptr);

        vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
        vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

        // Dispatch the compute job
        // The compute shader will do the frustum culling and adjust the indirect draw calls depending on object visibility.
        // It also determines the lod to use depending on distance to the viewer.
        vkCmdDispatch(compute.commandBuffer, objectCount / 64, 1, 1);

        // Add memory barrier to ensure that the compute shader has finished writing the indirect command buffer before it's consumed
        indirectCommandsBuffersBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        indirectCommandsBuffersBarriers[0].dstAccessMask = 0;
        indirectCommandsBuffersBarriers[0].buffer = indirectCommandsBuffer.buffer;
        indirectCommandsBuffersBarriers[0].size = indirectCommandsBuffer.descriptor.range;
        indirectCommandsBuffersBarriers[0].srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
        indirectCommandsBuffersBarriers[0].dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;

        indirectCommandsBuffersBarriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        indirectCommandsBuffersBarriers[1].dstAccessMask = 0;
        indirectCommandsBuffersBarriers[1].buffer = indirectDrawCountBuffer.buffer;
        indirectCommandsBuffersBarriers[1].size = indirectDrawCountBuffer.descriptor.range;
        indirectCommandsBuffersBarriers[1].srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
        indirectCommandsBuffersBarriers[1].dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;

        vkCmdPipelineBarrier(
            compute.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_FLAGS_NONE,
            0, nullptr,
            2, indirectCommandsBuffersBarriers,
            1, &depthImageReleaseBarrier);

        // todo: barrier for indirect stats buffer?

        vkEndCommandBuffer(compute.commandBuffer);
    }

    void setupDescriptorPool()
    {
        {
            std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2)
            };
            VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
            VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
        }
        std::vector<VkDescriptorPoolSize> poolSizesDepth = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 12),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12),
        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizesDepth, 12);
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &depthPyramidDescriptorPool));

    }

    void setupDescriptorSetLayout()
    {
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // Binding 0: Vertex shader uniform buffer
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,0),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,1),
            };
            VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

            VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));
        }
    }

    void setupDescriptorSet()
    {
        VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

        VkDescriptorBufferInfo vbDecriptor;
        vbDecriptor.buffer = lodModel.vertices.buffer;
        vbDecriptor.offset = 0;
        vbDecriptor.range = lodModel.vertices.count * sizeof(vkglTF::Vertex);

        std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
            // Binding 0: Vertex shader uniform buffer
            vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformData.scene.descriptor),
            vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &vbDecriptor),
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
    }

    void preparePipelines()
    {
        // This example uses two different input states, one for the instanced part and one for non-instanced rendering
        VkPipelineVertexInputStateCreateInfo inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
        std::vector<VkVertexInputBindingDescription> bindingDescriptions;
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions;

        // Vertex input bindings
        // The instancing pipeline uses a vertex input state with two bindings
        bindingDescriptions = {
            // Binding point 0: Mesh vertex layout description at per-vertex rate
            //vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, sizeof(vkglTF::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
            // Binding point 1: Instanced data at per-instance rate
            vks::initializers::vertexInputBindingDescription(INSTANCE_BUFFER_BIND_ID, sizeof(InstanceData), VK_VERTEX_INPUT_RATE_INSTANCE)
        };

        // Vertex attribute bindings
        attributeDescriptions = {
            // Per-vertex attributes
            // These are advanced for each vertex fetched by the vertex shader
           // vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(vkglTF::Vertex, pos)),	// Location 0: Position
           // vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(vkglTF::Vertex, normal)),	// Location 1: Normal
           // vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(vkglTF::Vertex, color)),	// Location 2: Texture coordinates
            // Per-Instance attributes
            // These are fetched for each instance rendered
            vks::initializers::vertexInputAttributeDescription(INSTANCE_BUFFER_BIND_ID, 4, VK_FORMAT_R32G32B32_SFLOAT, offsetof(InstanceData, pos)),	// Location 4: Position
            vks::initializers::vertexInputAttributeDescription(INSTANCE_BUFFER_BIND_ID, 5, VK_FORMAT_R32_SFLOAT, offsetof(InstanceData, scale)),		// Location 5: Scale
        };
        inputState.pVertexBindingDescriptions = bindingDescriptions.data();
        inputState.pVertexAttributeDescriptions = attributeDescriptions.data();
        inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
        inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
        VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS);
        VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
        VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);
        pipelineCreateInfo.pVertexInputState = &inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();

        // Indirect (and instanced) pipeline for the plants
        shaderStages[0] = loadShader(getShadersPath() + "computecullandlod/indirectdraw.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "computecullandlod/indirectdraw.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.plants));
    }

    void prepareBuffers()
    {
        objectCount = OBJECT_COUNT * OBJECT_COUNT * OBJECT_COUNT;

        vks::Buffer stagingBuffer;

        std::vector<InstanceData> instanceData(objectCount);
        indirectCommands.resize(objectCount);

        // Indirect draw commands
        for (uint32_t x = 0; x < OBJECT_COUNT; x++)
        {
            for (uint32_t y = 0; y < OBJECT_COUNT; y++)
            {
                for (uint32_t z = 0; z < OBJECT_COUNT; z++)
                {
                    uint32_t index = x + y * OBJECT_COUNT + z * OBJECT_COUNT * OBJECT_COUNT;
                    indirectCommands[index].instanceCount = 1;
                    indirectCommands[index].firstInstance = index;
                    // firstIndex and indexCount are written by the compute shader
                }
            }
        }

        indirectStats.drawCount = static_cast<uint32_t>(indirectCommands.size());

        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &stagingBuffer,
            indirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand),
            indirectCommands.data()));

        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &indirectCommandsBuffer,
            stagingBuffer.size));

        vulkanDevice->copyBuffer(&stagingBuffer, &indirectCommandsBuffer, queue);

        stagingBuffer.destroy();

        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &indirectDrawCountBuffer,
            sizeof(indirectStats)));

        // Map for host access
        VK_CHECK_RESULT(indirectDrawCountBuffer.map());

        // Instance data
        for (uint32_t x = 0; x < OBJECT_COUNT; x++)
        {
            for (uint32_t y = 0; y < OBJECT_COUNT; y++)
            {
                for (uint32_t z = 0; z < OBJECT_COUNT; z++)
                {
                    uint32_t index = x + y * OBJECT_COUNT + z * OBJECT_COUNT * OBJECT_COUNT;
                    instanceData[index].pos = glm::vec3((float)x, (float)y, (float)z) - glm::vec3((float)OBJECT_COUNT / 2.0f);
                    instanceData[index].scale = 2.0f;
                }
            }
        }

        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &stagingBuffer,
            instanceData.size() * sizeof(InstanceData),
            instanceData.data()));

        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &instanceBuffer,
            stagingBuffer.size));

        vulkanDevice->copyBuffer(&stagingBuffer, &instanceBuffer, queue);

        stagingBuffer.destroy();

        // Shader storage buffer containing index offsets and counts for the LODs
        struct LOD
        {
            uint32_t firstIndex;
            uint32_t indexCount;
            float distance;
            float _pad0;
        };
        std::vector<LOD> LODLevels;
        uint32_t n = 0;
        for (auto node : lodModel.nodes)
        {
            LOD lod;
            lod.firstIndex = node->mesh->primitives[0]->firstIndex;	// First index for this LOD
            lod.indexCount = node->mesh->primitives[0]->indexCount;	// Index count for this LOD
            lod.distance = 5.0f + n * 5.0f;							// Starting distance (to viewer) for this LOD
            n++;
            LODLevels.push_back(lod);
        }

        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &stagingBuffer,
            LODLevels.size() * sizeof(LOD),
            LODLevels.data()));

        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &compute.lodLevelsBuffers,
            stagingBuffer.size));

        vulkanDevice->copyBuffer(&stagingBuffer, &compute.lodLevelsBuffers, queue);

        stagingBuffer.destroy();

        // Scene uniform buffer
        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &uniformData.scene,
            sizeof(uboScene)));

        VK_CHECK_RESULT(uniformData.scene.map());

        updateUniformBuffer(true);
    }

    void prepareCompute()
    {
        prepareComputeDepthReduce();
        prepareComputeCull();
        // Build a single command buffer containing the compute dispatch commands
                // Separate command pool as queue family for compute may be different than graphics
        VkCommandPoolCreateInfo cmdPoolInfo = {};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

        // Create a command buffer for compute operations
        VkCommandBufferAllocateInfo cmdBufAllocateInfo =
            vks::initializers::commandBufferAllocateInfo(
                compute.commandPool,
                VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                1);

        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer));

        // Fence for compute CB sync
        VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &compute.fence));

        VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
        VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));
        buildComputeCommandBuffer();
    }

    struct SpecializationData
    {
        uint32_t maxLod;
        uint32_t objectCount;
    };

    void prepareComputeCull()
    {
        vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);

        // Create compute pipeline
        // Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0: Instance input data buffer
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0),
            // Binding 1: Indirect draw command output buffer (input)
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                1),
            // Binding 2: Uniform buffer with global matrices (input)
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                2),
            // Binding 3: Indirect draw stats (output)
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                3),
            // Binding 4: LOD info (input)
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                4),

            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                5),
        };

        VkDescriptorSetLayoutCreateInfo descriptorLayout =
            vks::initializers::descriptorSetLayoutCreateInfo(
                setLayoutBindings.data(),
                static_cast<uint32_t>(setLayoutBindings.size()));

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

        VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
            vks::initializers::pipelineLayoutCreateInfo(
                &compute.descriptorSetLayout,
                1);

        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

        VkDescriptorSetAllocateInfo allocInfo =
            vks::initializers::descriptorSetAllocateInfo(
                descriptorPool,
                &compute.descriptorSetLayout,
                1);

        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));

        auto depthPyramidDescriptor = vks::initializers::descriptorImageInfo(compute.depthSampler, depthPyramidImageView, VK_IMAGE_LAYOUT_GENERAL);

        std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
        {
            // Binding 0: Instance input data buffer
            vks::initializers::writeDescriptorSet(
                compute.descriptorSet,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                0,
                &instanceBuffer.descriptor),
            // Binding 1: Indirect draw command output buffer
            vks::initializers::writeDescriptorSet(
                compute.descriptorSet,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                1,
                &indirectCommandsBuffer.descriptor),
            // Binding 2: Uniform buffer with global matrices
            vks::initializers::writeDescriptorSet(
                compute.descriptorSet,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                2,
                &uniformData.scene.descriptor),
            // Binding 3: Atomic counter (written in shader)
            vks::initializers::writeDescriptorSet(
                compute.descriptorSet,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                3,
                &indirectDrawCountBuffer.descriptor),
            // Binding 4: LOD info
            vks::initializers::writeDescriptorSet(
                compute.descriptorSet,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                4,
                &compute.lodLevelsBuffers.descriptor),

            vks::initializers::writeDescriptorSet(
                compute.descriptorSet,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                5,
                &depthPyramidDescriptor),
        };

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

        // Create pipeline
        VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
        computePipelineCreateInfo.stage = loadShader(getShadersPath() + "computecullandlod/cull.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

        // Use specialization constants to pass max. level of detail (determined by no. of meshes)
        VkSpecializationMapEntry specializationEntries[2];
        specializationEntries[0].constantID = 0;
        specializationEntries[0].offset = offsetof(SpecializationData, maxLod);
        specializationEntries[0].size = sizeof(uint32_t);
        specializationEntries[1].constantID = 1;
        specializationEntries[1].offset = offsetof(SpecializationData, objectCount);
        specializationEntries[1].size = sizeof(uint32_t);

        SpecializationData specializationData = { static_cast<uint32_t>(lodModel.nodes.size()) - 1, objectCount };

        VkSpecializationInfo specializationInfo;
        specializationInfo.mapEntryCount = std::size(specializationEntries);
        specializationInfo.pMapEntries = specializationEntries;
        specializationInfo.dataSize = sizeof(specializationData);
        specializationInfo.pData = &specializationData;

        computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

        VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline));
    }

    void prepareComputeDepthReduce()
    {
        vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);

        // Create compute pipeline
        // Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {

            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0),

            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                1),
        };

        VkDescriptorSetLayoutCreateInfo descriptorLayout =
            vks::initializers::descriptorSetLayoutCreateInfo(
                setLayoutBindings.data(),
                static_cast<uint32_t>(setLayoutBindings.size()));

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.depthPyramidLevelDescriptorSetLayout));

        VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
            vks::initializers::pipelineLayoutCreateInfo(
                &compute.depthPyramidLevelDescriptorSetLayout,
                1);
        VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(ImageWidthPushConstant), 0);
        pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
        pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;

        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.depthPyramidPipelineLayout));
        std::vector<VkDescriptorSetLayout> layouts(depthLevelCount, compute.depthPyramidLevelDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo =
            vks::initializers::descriptorSetAllocateInfo(
                depthPyramidDescriptorPool,
                layouts.data(),
                depthLevelCount);
        compute.depthPyramidLevelDescriptorSets.resize(depthLevelCount);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.depthPyramidLevelDescriptorSets[0]));

        std::vector<VkDescriptorImageInfo> depthPyramidImageInfos;
        for (uint32_t level = 0; level < depthLevelCount; ++level)
        {
            const auto srcView = level == 0 ? depthStencil.view : depthPyramidLevelViews[level - 1];
            auto srcInfo = vks::initializers::descriptorImageInfo(compute.depthSampler, srcView, level == 0 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL);
            auto dstInfo = vks::initializers::descriptorImageInfo(nullptr, depthPyramidLevelViews[level], VK_IMAGE_LAYOUT_GENERAL);
            depthPyramidImageInfos.push_back(dstInfo);
            depthPyramidImageInfos.push_back(srcInfo);
        }
        std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets;
        for (uint32_t level = 0; level < depthLevelCount; ++level)
        {
            computeWriteDescriptorSets.push_back(
                vks::initializers::writeDescriptorSet(
                    compute.depthPyramidLevelDescriptorSets[level],
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0, &depthPyramidImageInfos[level * 2], 1));
            computeWriteDescriptorSets.push_back(
                vks::initializers::writeDescriptorSet(
                    compute.depthPyramidLevelDescriptorSets[level],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &depthPyramidImageInfos[level * 2 + 1], 1));
        };

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

        // Create pipeline
        VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.depthPyramidPipelineLayout, 0);
        computePipelineCreateInfo.stage = loadShader(getShadersPath() + "computecullandlod/depthreduce.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

        VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.depthPyramidPipeline));
    }

    void updateUniformBuffer(bool viewChanged)
    {
        if (viewChanged)
        {
            uboScene.projection = camera.matrices.perspective;
            uboScene.modelview = camera.matrices.view;
            if (!fixedFrustum)
            {
                uboScene.cameraPos = glm::vec4(camera.position, 1.0f) * -1.0f;
                frustum.update(uboScene.projection * uboScene.modelview);
                memcpy(uboScene.frustumPlanes, frustum.planes.data(), sizeof(glm::vec4) * 6);
            }
        }

        memcpy(uniformData.scene.mapped, &uboScene, sizeof(uboScene));
    }

    void draw()
    {
        VulkanExampleBase::prepareFrame();

        // Submit compute shader for frustum culling

        // Wait for fence to ensure that compute buffer writes have finished
        vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &compute.fence);

        VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
        computeSubmitInfo.signalSemaphoreCount = 1;
        computeSubmitInfo.pSignalSemaphores = &compute.semaphore;

        VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));

        // Submit graphics command buffer

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

        // Wait on present and compute semaphores
        std::array<VkPipelineStageFlags, 2> stageFlags = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        };
        std::array<VkSemaphore, 2> waitSemaphores = {
            semaphores.presentComplete,						// Wait for presentation to finished
            compute.semaphore								// Wait for compute to finish
        };

        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitDstStageMask = stageFlags.data();

        // Submit to queue
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, compute.fence));

        VulkanExampleBase::submitFrame();

        // Get draw count from compute
        memcpy(&indirectStats, indirectDrawCountBuffer.mapped, sizeof(indirectStats));
    }

    void prepare()
    {
        VulkanExampleBase::prepare();
        vkCmdDrawIndexedIndirectCountKHR = reinterpret_cast<PFN_vkCmdDrawIndexedIndirectCountKHR>(vkGetDeviceProcAddr(device, "vkCmdDrawIndexedIndirectCountKHR"));
        loadAssets();
        prepareBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        createDepthPyramidResources();
        prepareCompute();
        buildCommandBuffers();
        prepared = true;
    }

    virtual void render()
    {
        if (!prepared)
        {
            return;
        }
        draw();
        if (camera.updated)
        {
            updateUniformBuffer(true);
        }
    }

    virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay)
    {
        if (overlay->header("Settings")) {
            if (overlay->checkBox("Freeze frustum", &fixedFrustum)) {
                updateUniformBuffer(true);
            }
        }
        if (overlay->header("Statistics")) {
            overlay->text("Total objects: %d", objectCount);
            overlay->text("Visible objects: %d", indirectStats.drawCount);
            overlay->text("Visible tris: %d", indirectStats.primitiveCount);
            overlay->text("Occluded objects: %d", indirectStats.occluded);
            for (uint32_t i = 0; i < MAX_LOD_LEVEL + 1; i++) {
                overlay->text("LOD %d: %d", i, indirectStats.lodCount[i]);
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
