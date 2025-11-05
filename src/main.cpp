#include <format>
#include <iostream>
#include <array>
#include <string>
#include <fstream>
#include <cassert>
#include <stdexcept>

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "local_dirs.h"

namespace vkext
{

static PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = nullptr;

}

struct Buf
{
	VkBuffer buffer;
	VmaAllocation allocation;
	VmaAllocationInfo info;

	template<typename T>
	T* get_mapped_data()
	{
		assert(info.pMappedData);
		return reinterpret_cast<T*>(info.pMappedData);
	}
};

struct Img
{
    VkImage image;
    VkImageView image_view;
    VmaAllocation allocation;
};

class App
{
public:
	void run();

private:
	void init_vk();
	void cleanup();
	void create_pipeline();
	void create_commands();
    void create_background_image();
    void create_sampler();
    void create_output_image(uint32_t width, uint32_t height);
	void render(uint32_t width, uint32_t height);

    VkCommandBuffer begin_single_time_commands(
            VkQueue queue, VkCommandPool cmd_pool);
    void end_single_time_commands(
            VkQueue queue, VkCommandPool cmd_pool, VkCommandBuffer cmd_buffer);

private:
	VmaAllocator m_allocator;

	vkb::Instance m_instance;
	vkb::PhysicalDevice m_physical_device;
	vkb::Device m_device;
	VkQueue m_queue;
	VkDescriptorSetLayout m_desc_set_layout;
	VkPipelineLayout m_pipeline_layout;
	VkPipeline m_pipeline;

	VkCommandPool m_cmd_pool;
    VkSampler m_sampler;
    Img m_background_img;
    Img m_output_img;
};

namespace
{

std::vector<uint32_t> read_spv()
{
	std::ifstream file(std::string(vk_black_hole::get_shaders_dir()) + "black_hole.spv", std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open shader file");
	}
	size_t file_size = (size_t)file.tellg();
	assert(file_size % 4 == 0);
	std::vector<uint32_t> buffer(file_size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), file_size);
	file.close();
	return buffer;
}

Buf create_buffer(VmaAllocator allocator, VkDeviceSize size, bool host_access)
{
	VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0 };
	bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bci.size = size;

	VmaAllocationCreateInfo aci = {};
	aci.flags = host_access ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT : 0;
	aci.usage = VMA_MEMORY_USAGE_AUTO;

	VkBuffer buffer;
	VmaAllocation allocation;
	VmaAllocationInfo allocation_info;

	vmaCreateBuffer(allocator, &bci, &aci, &buffer, &allocation, &allocation_info);
	return { buffer, allocation, allocation_info };
}

}

void App::run()
{
	init_vk();
	create_pipeline();
	create_commands();
    create_background_image();
    create_output_image(1920, 1080);
    create_sampler();
    render(1920,1080);
	cleanup();
}

void App::cleanup()
{
    vkDestroyImageView(m_device.device, m_output_img.image_view, nullptr);
    vmaDestroyImage(m_allocator, m_output_img.image, m_output_img.allocation);
    vkDestroyImageView(m_device.device, m_background_img.image_view, nullptr);
    vmaDestroyImage(m_allocator, m_background_img.image, m_background_img.allocation);
    vkDestroySampler(m_device.device, m_sampler, nullptr);

	vkDestroyCommandPool(m_device.device, m_cmd_pool, nullptr);

	vkDestroyDescriptorSetLayout(m_device.device, m_desc_set_layout, nullptr);
	vkDestroyPipelineLayout(m_device.device, m_pipeline_layout, nullptr);
	vkDestroyPipeline(m_device.device, m_pipeline, nullptr);

	vmaDestroyAllocator(m_allocator);

	vkb::destroy_device(m_device);
	vkb::destroy_instance(m_instance);
}

void App::init_vk()
{
	vkb::InstanceBuilder builder;
	vkb::Instance instance = builder.
		set_app_name("vk black hole").
		set_headless(true).
		require_api_version(1, 3, 0).
		build().value();

	VkPhysicalDeviceMaintenance5FeaturesKHR m5f = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR, 0 };
	m5f.maintenance5 = VK_TRUE;
	//VkPhysicalDeviceFeatures2 f = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, 0 };
	//f.pNext = &m5f;
#if 0
	VkPhysicalDeviceVulkan11Features v11f = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, 0 };
	v11f.storageBuffer16BitAccess = VK_TRUE;
	VkPhysicalDeviceVulkan12Features v12f = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, 0 };
	v12f.shaderFloat16 = VK_TRUE;
	v12f.descriptorBindingPartiallyBound = VK_TRUE;
#endif
	VkPhysicalDeviceVulkan13Features v13f = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, 0 };
	v13f.synchronization2 = VK_TRUE;
    v13f.subgroupSizeControl = VK_TRUE;
    v13f.computeFullSubgroups = VK_TRUE;

	vkb::PhysicalDeviceSelector selector{ instance };
	vkb::PhysicalDevice phys_dev = selector.
//		prefer_gpu_device_type(vkb::PreferredDeviceType::integrated).
		set_minimum_version(1, 3).
//		set_required_features_11(v11f).
//		set_required_features_12(v12f).
		set_required_features_13(v13f).
		add_required_extension_features(m5f).
		add_required_extensions({
			VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, 
			VK_KHR_MAINTENANCE_5_EXTENSION_NAME, 
		}).
		select().value();

	vkb::DeviceBuilder device_builder{ phys_dev };
	vkb::Device device = device_builder.
		build().value();

	std::cout << std::format("Selected {}", device.physical_device.name) << std::endl;

    vkext::vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)instance.fp_vkGetDeviceProcAddr(device.device, "vkCmdPushDescriptorSetKHR");

	VmaAllocatorCreateInfo vci = {};
	vci.vulkanApiVersion = VK_API_VERSION_1_3;
	vci.physicalDevice = phys_dev.physical_device;
	vci.device = device.device;
	vci.instance = instance.instance;
	vmaCreateAllocator(&vci, &m_allocator);

	m_instance = instance;
	m_physical_device = phys_dev;
	m_device = device;
	m_queue = device.get_queue(vkb::QueueType::graphics).value();
}

void App::create_pipeline()
{
	auto create_binding = [](uint32_t binding, VkDescriptorType type) {
		VkDescriptorSetLayoutBinding b = {};
		b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		b.descriptorType = type;
		b.binding = binding;
		b.descriptorCount = 1;
		return b;
	};
	
	std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
		create_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
		create_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
	};
	VkDescriptorSetLayoutCreateInfo dslci = {};
	dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dslci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
	dslci.bindingCount = bindings.size();
	dslci.pBindings = bindings.data();

	vkCreateDescriptorSetLayout(m_device.device, &dslci, nullptr, &m_desc_set_layout);

	VkPipelineLayoutCreateInfo plci = {};
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.setLayoutCount = 1;
	plci.pSetLayouts = &m_desc_set_layout;

	vkCreatePipelineLayout(m_device.device, &plci, nullptr, &m_pipeline_layout);

	std::vector<uint32_t> shader_code = read_spv();
	VkShaderModuleCreateInfo shader_module = {};
	shader_module.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shader_module.codeSize = shader_code.size() * sizeof(uint32_t);
	shader_module.pCode = shader_code.data();

#if 0
    VkSpecializationMapEntry sme = {};
    sme.constantID = 0;
    sme.offset = 0;
    sme.size = sizeof(uint32_t);

    VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, 0 };
	VkPhysicalDeviceSubgroupSizeControlProperties subgroup_props = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, 0 };
	props2.pNext = &subgroup_props;
	vkGetPhysicalDeviceProperties2(m_physical_device.physical_device, &props2);

    VkSpecializationInfo specialization = {};
    specialization.mapEntryCount = 1;
    specialization.pMapEntries = &sme;
    specialization.dataSize = sizeof(uint32_t);
    specialization.pData = &subgroup_props.minSubgroupSize;
#endif

	VkPipelineShaderStageCreateInfo ssci = {};
	ssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	ssci.pName = "main";
    ssci.pNext = &shader_module;
#if 0
    ssci.pSpecializationInfo = &specialization;
    ssci.flags =
        VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT |
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
#endif

	VkComputePipelineCreateInfo pci = {};
	pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pci.stage = ssci;
	pci.layout = m_pipeline_layout;

	vkCreateComputePipelines(m_device.device, VK_NULL_HANDLE, 1, &pci, nullptr, &m_pipeline);
}

void App::create_commands()
{
	VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0 };
	cpci.queueFamilyIndex = m_device.get_queue_index(vkb::QueueType::graphics).value();
	cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	vkCreateCommandPool(m_device.device, &cpci, nullptr, &m_cmd_pool);
}

void App::create_background_image()
{
    std::string img_path = std::string(vk_black_hole::get_resources_dir()) + "esa.png";
    int width, height, channels;
    stbi_uc *img_data = stbi_load(img_path.c_str(), &width, &height, &channels, 4);
    Buf buf = create_buffer(m_allocator, 4*width*height, true);
    memcpy(buf.get_mapped_data<uint8_t>(), img_data, 4*width*height);
    stbi_image_free(img_data);

    VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, 0 };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = { 
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1u
    };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo aci = {};
	aci.usage = VMA_MEMORY_USAGE_AUTO;
    Img img;
    vmaCreateImage(m_allocator, &ici, &aci, &img.image, &img.allocation, nullptr);

    VkImageViewCreateInfo ivci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, 0 };
    ivci.image = img.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.components = { 
        VK_COMPONENT_SWIZZLE_IDENTITY, 
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY
    };
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(m_device.device, &ivci, nullptr, &img.image_view);

    // copy from buffer to image
    auto cmd_buf = begin_single_time_commands(m_queue, m_cmd_pool);

    {
        VkImageMemoryBarrier2 bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, 0 };
        bar.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        bar.srcAccessMask = VK_ACCESS_2_NONE;
        bar.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        bar.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.image = img.image;
        bar.subresourceRange = ivci.subresourceRange;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO, 0 };
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;
        vkCmdPipelineBarrier2(cmd_buf, &dep);
    }

    VkBufferImageCopy cpy = {};
    cpy.bufferOffset = 0;
    cpy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    cpy.imageSubresource.mipLevel = 0;
    cpy.imageSubresource.baseArrayLayer = 0;
    cpy.imageSubresource.layerCount = 1;
    cpy.imageExtent = { 
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1
    };
    vkCmdCopyBufferToImage(cmd_buf, buf.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cpy);

    {
        VkImageMemoryBarrier2 bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, 0 };
        bar.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        bar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bar.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        bar.dstAccessMask = VK_ACCESS_2_NONE;
        bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
        bar.image = img.image;
        bar.subresourceRange = ivci.subresourceRange;

        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO, 0 };
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;
        vkCmdPipelineBarrier2(cmd_buf, &dep);
    }

    end_single_time_commands(m_queue, m_cmd_pool, cmd_buf);

    m_background_img = img;
    vmaDestroyBuffer(m_allocator, buf.buffer, buf.allocation);
}

void App::create_output_image(uint32_t width, uint32_t height)
{
    VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, 0 };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {  width, height, 1u };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo aci = {};
	aci.usage = VMA_MEMORY_USAGE_AUTO;
    Img img;
    vmaCreateImage(m_allocator, &ici, &aci, &img.image, &img.allocation, nullptr);

    VkImageViewCreateInfo ivci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, 0 };
    ivci.image = img.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.components = { 
        VK_COMPONENT_SWIZZLE_IDENTITY, 
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY
    };
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(m_device.device, &ivci, nullptr, &img.image_view);

    m_output_img = img;
}

void App::create_sampler()
{
    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, 0 };
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.minLod = 0;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    vkCreateSampler(m_device.device, &sci, nullptr, &m_sampler);
}

void App::render(uint32_t width, uint32_t height)
{
	Buf output_buf = create_buffer(m_allocator, 4*width*height, true);

	vkResetCommandPool(m_device.device, m_cmd_pool, 0);
    auto cmd_buf = begin_single_time_commands(m_queue, m_cmd_pool);

    {
        VkImageMemoryBarrier2 bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, 0 };
        bar.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        bar.srcAccessMask = VK_ACCESS_2_NONE;
        bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bar.image = m_output_img.image;
        bar.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo dep = {};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;
        vkCmdPipelineBarrier2(cmd_buf, &dep);
    }

	vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

	std::array<VkWriteDescriptorSet, 2> desc_writes = {};
	std::array<VkDescriptorImageInfo, 2> image_desc_writes = {};
    {
        image_desc_writes[0].imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
        image_desc_writes[0].imageView = m_background_img.image_view;
        image_desc_writes[0].sampler = m_sampler;
        desc_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        desc_writes[0].dstBinding = 0;
        desc_writes[0].dstArrayElement = 0;
        desc_writes[0].descriptorCount = 1;
        desc_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        desc_writes[0].pImageInfo = &image_desc_writes[0];
    }
    {
        image_desc_writes[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_desc_writes[1].imageView = m_output_img.image_view;
        image_desc_writes[1].sampler = VK_NULL_HANDLE;
        desc_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        desc_writes[1].dstBinding = 1;
        desc_writes[1].dstArrayElement = 0;
        desc_writes[1].descriptorCount = 1;
        desc_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        desc_writes[1].pImageInfo = &image_desc_writes[1];
    }

    vkext::vkCmdPushDescriptorSetKHR(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 0,
		desc_writes.size(), desc_writes.data());
	vkCmdDispatch(cmd_buf, width/8, height/8, 1);

    {
        VkImageMemoryBarrier2 bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, 0 };
        bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        bar.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        bar.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        bar.image = m_output_img.image;
        bar.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo dep = {};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;
        vkCmdPipelineBarrier2(cmd_buf, &dep);
    }

    VkBufferImageCopy cpy = {};
    cpy.bufferOffset = 0;
    cpy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    cpy.imageExtent = { width, height, 1 };
    vkCmdCopyImageToBuffer(cmd_buf, m_output_img.image, 
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, output_buf.buffer, 1, &cpy);

    end_single_time_commands(m_queue, m_cmd_pool, cmd_buf);

	//vkDeviceWaitIdle(m_device.device);
    
    stbi_write_png("out.png", width, height, 4, output_buf.get_mapped_data<uint8_t>(), width*4);

	vmaDestroyBuffer(m_allocator, output_buf.buffer, output_buf.allocation);
}

VkCommandBuffer App::begin_single_time_commands(VkQueue queue, VkCommandPool cmd_pool)
{
	VkCommandBufferAllocateInfo ai = {};
	ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandPool = cmd_pool;
	ai.commandBufferCount = 1;

	VkCommandBuffer cmd_buf;
	auto res = vkAllocateCommandBuffers(m_device, &ai, &cmd_buf);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate command buffer");
	VkCommandBufferBeginInfo bi = {};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	res = vkBeginCommandBuffer(cmd_buf, &bi);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to begin command buffer recording");
	return cmd_buf;
}

void App::end_single_time_commands(VkQueue queue, VkCommandPool cmd_pool, VkCommandBuffer cmd_buffer)
{
	auto res = vkEndCommandBuffer(cmd_buffer);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to end command buffer recording");

	VkCommandBufferSubmitInfoKHR cmd_submit = {};
	cmd_submit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR;
	cmd_submit.commandBuffer = cmd_buffer;
	cmd_submit.deviceMask = 0;

	VkSubmitInfo2 submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR;
	submit_info.commandBufferInfoCount = 1;
	submit_info.pCommandBufferInfos = &cmd_submit;

	res = vkQueueSubmit2(queue, 1, &submit_info, VK_NULL_HANDLE);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to submit to queue");
	res  = vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(m_device, cmd_pool, 1, &cmd_buffer);
}

int main()
{
	App app;
	app.run();
	return 0;
}
