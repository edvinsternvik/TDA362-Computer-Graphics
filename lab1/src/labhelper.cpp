#include "labhelper.hpp"
#include <set>
#include <fstream>

#ifdef NDEBUG
const std::array<const char*, 0> VALIDATION_LAYERS = { };
#else
const std::array<const char*, 1> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};
#endif

const std::array<const char*, 1> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

void init_vulkan(
    SDL_Window* window,
    VkInstance* vk_instance,
    VkDevice* vk_device, VkPhysicalDevice* physical_device,
    uint32_t* graphics_family, uint32_t* present_family,
    VkQueue* graphics_queue, VkQueue* present_queue,
    VkSurfaceKHR* surface
) {
    // Check for validation layers
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    bool validation_layers_found = true;
    for(const char* validation_layer : VALIDATION_LAYERS) {
        bool layer_found = false;
        for(const auto& layer : available_layers) {
            if(std::strcmp(validation_layer, layer.layerName) == 0) {
                layer_found = true;
                break;
            }
        }

        if(!layer_found) {
            validation_layers_found = false;
            break;
        }
    }
    if(!VALIDATION_LAYERS.empty() && !validation_layers_found) {
        throw std::runtime_error("Could not find validation layers");
    }

    // Get vulkan instance
    uint32_t inst_ext_count;
    SDL_Vulkan_GetInstanceExtensions(window, &inst_ext_count, nullptr);
    std::vector<const char*> inst_ext_names(inst_ext_count);
    SDL_Vulkan_GetInstanceExtensions(window, &inst_ext_count, inst_ext_names.data());

    VkInstanceCreateInfo inst_create_info = {};
    inst_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_create_info.enabledExtensionCount = inst_ext_names.size();
    inst_create_info.ppEnabledExtensionNames = inst_ext_names.data();
    inst_create_info.enabledLayerCount = VALIDATION_LAYERS.size();
    inst_create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();

    VK_HANDLE_ERROR(
        vkCreateInstance(&inst_create_info, nullptr, vk_instance),
        "Could not create instance"
    );

    // Create surface
    SDL_Vulkan_CreateSurface(window, *vk_instance, surface);

    // Get physical devices
    uint32_t phys_device_count;
    vkEnumeratePhysicalDevices(*vk_instance, &phys_device_count, nullptr);
    std::vector<VkPhysicalDevice> physical_devices(phys_device_count);
    vkEnumeratePhysicalDevices(*vk_instance, &phys_device_count, physical_devices.data());

    for(const auto curr_phys_device : physical_devices) {
        // Get queue families
        uint32_t queue_family_count;
        vkGetPhysicalDeviceQueueFamilyProperties(curr_phys_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(curr_phys_device, &queue_family_count, queue_families.data());

        std::optional<uint32_t> current_graphics_family;
        std::optional<uint32_t> current_present_family;

        for(uint32_t i = 0; i < queue_families.size(); ++i) {
            if(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) current_graphics_family = i;

            VkBool32 present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(curr_phys_device, i, *surface, &present_support);
            if(present_support) current_present_family = i;
        }
        if(!current_graphics_family.has_value() || !current_present_family.has_value()) continue;

        // Check for extension support
        uint32_t extension_count;
        vkEnumerateDeviceExtensionProperties(curr_phys_device, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> available_extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(curr_phys_device, nullptr, &extension_count, available_extensions.data());

        bool extensions_supported = true;
        for(const char* extension : DEVICE_EXTENSIONS) {
            bool extension_supported = false;
            for(const auto& available_extension : available_extensions) {
                if(strcmp(extension, available_extension.extensionName) == 0) {
                    extension_supported = true;
                    break;
                }
            }
            if(!extension_supported) extensions_supported = false;
        }
        if(!extensions_supported) continue;

        // TODO
        bool valid_swapchain = true;
        if(!valid_swapchain) continue;

        // Check for supported features
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(curr_phys_device, &features);
        if(!features.samplerAnisotropy) continue;

        *graphics_family = current_graphics_family.value();
        *present_family = current_present_family.value();
        *physical_device = curr_phys_device;
        break;
    }

    std::set<uint32_t> unique_queue_families = {graphics_family, present_family};

    // Get logical device
    float queue_priorities[1] = { 1.0 };
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    for(uint32_t queue_family : unique_queue_families) {
        VkDeviceQueueCreateInfo queue_create_info = {};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = queue_priorities;

        queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures device_features = {};
    device_features.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.enabledExtensionCount = DEVICE_EXTENSIONS.size();
    device_create_info.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    device_create_info.queueCreateInfoCount = queue_create_infos.size();
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.pEnabledFeatures = &device_features;

    VK_HANDLE_ERROR(
        vkCreateDevice(*physical_device, &device_create_info, nullptr, vk_device),
        "Could not create device"
    );

    // Get queues
    vkGetDeviceQueue(*vk_device, *graphics_family, 0, graphics_queue);
    vkGetDeviceQueue(*vk_device, *present_family, 0, present_queue);
}

SurfaceInfo get_surface_info(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surface_capabilities);

    uint32_t surface_format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> surface_formats(surface_format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, surface_formats.data());

    uint32_t present_modes_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_modes_count, nullptr);
    std::vector<VkPresentModeKHR> present_modes(present_modes_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_modes_count, present_modes.data());

    // TODO - Select optimal format, present mode and swap extent
    SurfaceInfo surface_info = {};
    surface_info.format = surface_formats[0];
    surface_info.present_mode = VK_PRESENT_MODE_FIFO_KHR;
    surface_info.capabilities = surface_capabilities;

    return surface_info;
}

std::vector<char> read_file(const char* file_name) {
    std::ifstream fs(file_name, std::ios::ate | std::ios::binary);
    size_t file_size = fs.tellg();
    std::vector<char> file_buffer(file_size);
    fs.seekg(0);
    fs.read(file_buffer.data(), file_size);
    return file_buffer;
}

VkShaderModule create_shader_module(VkDevice device, std::vector<char>& shader_src) {
    VkShaderModuleCreateInfo shader_module_create_info = {};
    shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_create_info.codeSize = shader_src.size();
    shader_module_create_info.pCode = reinterpret_cast<const uint32_t*>(shader_src.data());

    VkShaderModule shader_module;
    VK_HANDLE_ERROR(
        vkCreateShaderModule(device, &shader_module_create_info, nullptr, &shader_module),
        "Could not create shader module"
    );
    return shader_module;
}

VkCommandBuffer begin_single_use_command(
    VkDevice device,
    VkCommandPool command_pool
) {
    VkCommandBuffer cmd_buffer;
    VkCommandBufferAllocateInfo cmd_buffer_allocate_info = {};
    cmd_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_buffer_allocate_info.commandPool = command_pool;
    cmd_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_buffer_allocate_info.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmd_buffer_allocate_info, &cmd_buffer);

    VkCommandBufferBeginInfo cmd_begin_info = {};
    cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buffer, &cmd_begin_info);

    return cmd_buffer;
}

void submit_command(
    VkQueue cmd_queue,
    VkCommandBuffer cmd_buffer
) {
    vkEndCommandBuffer(cmd_buffer);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;
    vkQueueSubmit(cmd_queue, 1, &submit_info, VK_NULL_HANDLE);
}

VkBuffer create_buffer(
    VkDevice device,
    VkBufferUsageFlags usage,
    size_t buffer_size
) {
    VkBufferCreateInfo buffer_create_info = {};
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.size = buffer_size;
    buffer_create_info.usage = usage;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VK_HANDLE_ERROR(
        vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer),
        "Could not create vertex buffer"
    );

    return buffer;
}

uint32_t get_suitable_memory_type_index(
    VkPhysicalDevice physical_device,
    VkMemoryRequirements mem_requirements,
    VkMemoryPropertyFlags properties
) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    uint32_t mem_type = -1;
    for(uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        if(
            (mem_requirements.memoryTypeBits & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties
        ) {
            mem_type = i;
            break;
        }
    }
    if(mem_type == -1) throw std::runtime_error("Could not find suitable memory type");

    return mem_type;
}

VkDeviceMemory allocate_buffer_memory(
    VkDevice device, VkPhysicalDevice physical_device,
    VkMemoryPropertyFlags properties,
    VkBuffer buffer
) {
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = get_suitable_memory_type_index(physical_device, mem_requirements, properties);

    VkDeviceMemory buffer_memory;
    VK_HANDLE_ERROR(
        vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory),
        "Could not allocate buffer memory"
    );
    vkBindBufferMemory(device, buffer, buffer_memory, 0);

    return buffer_memory;
}

void write_buffer_staged(
    VkDevice device, VkPhysicalDevice physical_device, VkQueue cmd_queue,
    VkCommandPool command_pool,
    VkBuffer buffer,
    void* data, size_t data_size
) {
    VkBuffer staging_buffer = create_buffer(
        device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data_size
    );

    VkDeviceMemory staging_buffer_memory = allocate_buffer_memory(
        device, physical_device,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging_buffer
    );

    void* buffer_data;
    vkMapMemory(device, staging_buffer_memory, 0, data_size, 0, &buffer_data);
    memcpy(buffer_data, data, data_size);
    vkUnmapMemory(device, staging_buffer_memory);

    VkCommandBuffer cmd_buffer = begin_single_use_command(device, command_pool);

    VkBufferCopy buffer_cpy = {};
    buffer_cpy.srcOffset = 0;
    buffer_cpy.dstOffset = 0;
    buffer_cpy.size = data_size;
    vkCmdCopyBuffer(cmd_buffer, staging_buffer, buffer, 1, &buffer_cpy);

    submit_command(cmd_queue, cmd_buffer);

    vkDeviceWaitIdle(device);

    vkFreeCommandBuffers(device, command_pool, 1, &cmd_buffer);
    vkDestroyBuffer(device, staging_buffer, nullptr);
    vkFreeMemory(device, staging_buffer_memory, nullptr);
}

VkSwapchainKHR create_swapchain(
    VkDevice device,
    VkSurfaceKHR surface,
    SurfaceInfo surface_info,
    uint32_t graphics_family,
    uint32_t present_family
) {
    uint32_t swapchain_min_image_count = surface_info.capabilities.minImageCount + 1;

    VkSwapchainCreateInfoKHR swapchain_create_info = {};
    swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_create_info.surface = surface;
    swapchain_create_info.minImageCount = swapchain_min_image_count;
    swapchain_create_info.imageFormat = surface_info.format.format;
    swapchain_create_info.imageColorSpace = surface_info.format.colorSpace;
    swapchain_create_info.imageExtent = surface_info.capabilities.currentExtent;
    swapchain_create_info.imageArrayLayers = 1;
    swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchain_create_info.preTransform = surface_info.capabilities.currentTransform;
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_create_info.presentMode = surface_info.present_mode;
    swapchain_create_info.clipped = VK_TRUE;
    swapchain_create_info.oldSwapchain = VK_NULL_HANDLE;

    std::array<uint32_t, 2> queue_family_indices = { graphics_family, present_family };
    if(graphics_family == present_family) {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchain_create_info.queueFamilyIndexCount = 0;
        swapchain_create_info.pQueueFamilyIndices = nullptr;
    }
    else {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_create_info.queueFamilyIndexCount = 2;
        swapchain_create_info.pQueueFamilyIndices = queue_family_indices.data();
    }

    VkSwapchainKHR swapchain;
    VK_HANDLE_ERROR(
        vkCreateSwapchainKHR(device, &swapchain_create_info, nullptr, &swapchain),
        "Could not create swapchain"
    );

    return swapchain;
}

std::vector<VkImageView> create_swapchain_image_views(
    VkDevice device,
    VkSwapchainKHR swapchain,
    VkFormat image_format
) {
    uint32_t swapchain_image_count;
    vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr);
    std::vector<VkImage> swapchain_images(swapchain_image_count);
    vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images.data());

    std::vector<VkImageView> swapchain_image_views(swapchain_image_count);
    for(size_t i = 0; i < swapchain_image_count; ++i) {
        VkImageViewCreateInfo view_create_info = {};
        view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_create_info.image = swapchain_images[i];
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_create_info.format = image_format;
        view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_create_info.subresourceRange.baseMipLevel = 0;
        view_create_info.subresourceRange.levelCount = 1;
        view_create_info.subresourceRange.baseArrayLayer = 0;
        view_create_info.subresourceRange.layerCount = 1;

        VK_HANDLE_ERROR(
            vkCreateImageView(device, &view_create_info, nullptr, &swapchain_image_views[i]),
            "Could not create image view"
        );
    }

    return swapchain_image_views;
}

VkDeviceMemory allocate_image_memory(
    VkDevice device, VkPhysicalDevice physical_device,
    VkMemoryPropertyFlags properties,
    VkImage image
) {
    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device, image, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = get_suitable_memory_type_index(physical_device, mem_requirements, properties);

    VkDeviceMemory image_memory;
    VK_HANDLE_ERROR(
        vkAllocateMemory(device, &alloc_info, nullptr, &image_memory),
        "Could not allocate buffer memory"
    );
    vkBindImageMemory(device, image, image_memory, 0);

    return image_memory;
}

void transition_image_layout(
    VkDevice device,
    VkCommandPool command_pool,
    VkQueue command_queue,
    VkImage image,
    VkFormat format,
    VkImageLayout old_layout, VkImageLayout new_layout
) {
    VkCommandBuffer cmd_buffer = begin_single_use_command(device, command_pool);

    auto get_access_flag = [](VkImageLayout layout) {
        switch(layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED: return VK_ACCESS_NONE;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return VK_ACCESS_SHADER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return VK_ACCESS_TRANSFER_WRITE_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return VkAccessFlagBits(
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
        );
        default: throw std::runtime_error("Unsupported image layout transition");
        }
    };

    auto get_stage_flag = [](VkImageLayout layout) {
        switch(layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED: return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return VK_PIPELINE_STAGE_TRANSFER_BIT;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        default: throw std::runtime_error("Unsupported image layout transition");
        }
    };

    auto get_aspect = [](VkImageLayout layout, VkFormat format) {
        VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
        if(layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            aspect_mask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if(format == VK_FORMAT_D32_SFLOAT_S8_UINT) aspect_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            if(format == VK_FORMAT_D24_UNORM_S8_UINT) aspect_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        return aspect_mask;
    };

    VkAccessFlags src_access = get_access_flag(old_layout);
    VkAccessFlags dst_access = get_access_flag(new_layout);
    VkPipelineStageFlags src_stage = get_stage_flag(old_layout);
    VkPipelineStageFlags dst_stage = get_stage_flag(new_layout);
    VkImageAspectFlags aspect_mask = get_aspect(new_layout, format);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = aspect_mask;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.baseArrayLayer = 0;
    vkCmdPipelineBarrier(
        cmd_buffer,
        src_stage, dst_stage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    submit_command(command_queue, cmd_buffer);

    vkDeviceWaitIdle(device);
    vkFreeCommandBuffers(device, command_pool, 1, &cmd_buffer);
}

void create_depth_buffer(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    uint32_t width, uint32_t height,
    VkImage* depth_image,
    VkDeviceMemory* depth_image_memory,
    VkImageView* depth_image_view
) {
    VkImageCreateInfo depth_image_create_info = {};
    depth_image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depth_image_create_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depth_image_create_info.extent.width = width;
    depth_image_create_info.extent.height = height;
    depth_image_create_info.extent.depth = 1;
    depth_image_create_info.format = VK_FORMAT_D32_SFLOAT; // TODO: select supported format
    depth_image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    depth_image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_image_create_info.imageType = VK_IMAGE_TYPE_2D;
    depth_image_create_info.mipLevels = 1;
    depth_image_create_info.arrayLayers = 1;
    depth_image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    depth_image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_image_create_info.queueFamilyIndexCount = 0;
    vkCreateImage(device, &depth_image_create_info, nullptr, depth_image);

    *depth_image_memory = allocate_image_memory(
        device,
        physical_device,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        *depth_image
    );

    VkImageViewCreateInfo depth_view_create_info = {};
    depth_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depth_view_create_info.image = *depth_image;
    depth_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depth_view_create_info.format = VK_FORMAT_D32_SFLOAT;
    depth_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_view_create_info.subresourceRange.baseMipLevel = 0;
    depth_view_create_info.subresourceRange.levelCount = 1;
    depth_view_create_info.subresourceRange.baseArrayLayer = 0;
    depth_view_create_info.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &depth_view_create_info, nullptr, depth_image_view);

    transition_image_layout(
        device,
        command_pool,
        command_queue,
        *depth_image,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );
}

VkFramebuffer create_framebuffer(
    VkDevice device,
    VkRenderPass render_pass,
    VkImageView image_view,
    VkImageView depth_view,
    VkExtent2D extent
) {
    std::array<VkImageView, 2> attachments = {
        image_view, depth_view
    };

    VkFramebufferCreateInfo framebuffer_create_info = {};
    framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_create_info.renderPass = render_pass;
    framebuffer_create_info.attachmentCount = attachments.size();
    framebuffer_create_info.pAttachments = attachments.data();
    framebuffer_create_info.width = extent.width;
    framebuffer_create_info.height = extent.height;
    framebuffer_create_info.layers = 1;

    VkFramebuffer framebuffer;
    VK_HANDLE_ERROR(
        vkCreateFramebuffer(device, &framebuffer_create_info, nullptr, &framebuffer),
        "Could not create framebuffer"
    );

    return framebuffer;
}

void recreate_swapchain(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    VkSwapchainKHR* swapchain,
    std::vector<VkImageView>* image_views,
    VkImage* depth_image,
    VkDeviceMemory* depth_image_memory,
    VkImageView* depth_image_view,
    std::vector<VkFramebuffer>* framebuffers,
    VkSurfaceKHR surface,
    SurfaceInfo surface_info,
    uint32_t graphics_family,
    uint32_t present_family,
    VkRenderPass render_pass
) {
    if(*swapchain != VK_NULL_HANDLE) {
        for(const auto fb : *framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        for(const auto iv : *image_views) vkDestroyImageView(device, iv, nullptr);
        vkDestroyImageView(device, *depth_image_view, nullptr);
        vkDestroyImage(device, *depth_image, nullptr);
        vkFreeMemory(device, *depth_image_memory, nullptr);
        vkDestroySwapchainKHR(device, *swapchain, nullptr);
    }

    *swapchain = create_swapchain(device, surface, surface_info, graphics_family, present_family);
    *image_views = create_swapchain_image_views(device, *swapchain, surface_info.format.format);
    create_depth_buffer(
        device, physical_device,
        command_pool, command_queue,
        surface_info.capabilities.currentExtent.width,
        surface_info.capabilities.currentExtent.height,
        depth_image, depth_image_memory, depth_image_view
    );
    *framebuffers = std::vector<VkFramebuffer>(image_views->size());
    for(size_t i = 0; i < image_views->size(); ++i) {
        (*framebuffers)[i] = create_framebuffer(
            device,
            render_pass,
            (*image_views)[i], *depth_image_view,
            surface_info.capabilities.currentExtent
        );
    }
}

void write_image_staged(
    VkDevice device, VkPhysicalDevice physical_device, VkQueue cmd_queue,
    VkCommandPool command_pool,
    VkImage image,
    void* data, size_t data_size, uint32_t width, uint32_t height
) {
    VkBuffer staging_buffer = create_buffer(
        device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data_size
    );

    VkDeviceMemory staging_buffer_memory = allocate_buffer_memory(
        device, physical_device,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging_buffer
    );

    void* buffer_data;
    vkMapMemory(device, staging_buffer_memory, 0, data_size, 0, &buffer_data);
    memcpy(buffer_data, data, data_size);
    vkUnmapMemory(device, staging_buffer_memory);

    VkCommandBuffer cmd_buffer = begin_single_use_command(device, command_pool);

    VkBufferImageCopy image_copy_region = {};
    image_copy_region.imageExtent.width = width;
    image_copy_region.imageExtent.height = height;
    image_copy_region.imageExtent.depth = 1;
    image_copy_region.imageOffset = { 0, 0, 0 };
    image_copy_region.bufferOffset = 0;
    image_copy_region.bufferRowLength = 0;
    image_copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_copy_region.imageSubresource.layerCount = 1;
    image_copy_region.imageSubresource.baseArrayLayer = 0;
    image_copy_region.imageSubresource.mipLevel = 0;
    vkCmdCopyBufferToImage(
        cmd_buffer,
        staging_buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &image_copy_region
    );

    submit_command(cmd_queue, cmd_buffer);

    vkDeviceWaitIdle(device);

    vkFreeCommandBuffers(device, command_pool, 1, &cmd_buffer);
    vkDestroyBuffer(device, staging_buffer, nullptr);
    vkFreeMemory(device, staging_buffer_memory, nullptr);
}
