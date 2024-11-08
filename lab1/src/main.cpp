#include <SDL2/SDL.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <vector>
#include <set>
#include <iostream>
#include <vulkan/vulkan_core.h>
#include <array>
#include <fstream>

const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;
const std::array<const char*, 1> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

#ifdef NDEBUG
const std::array<const char*, 0> VALIDATION_LAYERS = { };
#else
const std::array<const char*, 1> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};
#endif

#define VK_HANDLE_ERROR(X, msg) if(X != VK_SUCCESS) { throw std::runtime_error(msg); }

int main() {
    // Init
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Vulkan_LoadLibrary(nullptr);

    // Create window
    SDL_Window* window = SDL_CreateWindow(
        "Vulkan test",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        SCREEN_WIDTH, SCREEN_HEIGHT,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN
    );

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
    if(validation_layers_found && !VALIDATION_LAYERS.empty()) {
        std::cout << "Using validation layers" << std::endl;
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

    VkInstance vk_instance;
    VK_HANDLE_ERROR(
        vkCreateInstance(&inst_create_info, nullptr, &vk_instance),
        "Could not create instance"
    );

    // Create surface
    VkSurfaceKHR surface;
    SDL_Vulkan_CreateSurface(window, vk_instance, &surface);

    // Get physical devices
    uint32_t phys_device_count;
    vkEnumeratePhysicalDevices(vk_instance, &phys_device_count, nullptr);
    std::vector<VkPhysicalDevice> physical_devices(phys_device_count);
    vkEnumeratePhysicalDevices(vk_instance, &phys_device_count, physical_devices.data());

    uint32_t graphics_family = 0;
    uint32_t present_family = 0;
    VkPhysicalDevice physical_device = nullptr;
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
            vkGetPhysicalDeviceSurfaceSupportKHR(curr_phys_device, i, surface, &present_support);
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

        graphics_family = current_graphics_family.value();
        present_family = current_present_family.value();
        physical_device = curr_phys_device;
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

    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.enabledExtensionCount = DEVICE_EXTENSIONS.size();
    device_create_info.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    device_create_info.queueCreateInfoCount = queue_create_infos.size();
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.pEnabledFeatures = &device_features;

    VkDevice vk_device;
    VK_HANDLE_ERROR(
        vkCreateDevice(physical_device, &device_create_info, nullptr, &vk_device),
        "Could not create device"
    );

    // Get queues
    VkQueue graphics_queue;
    vkGetDeviceQueue(vk_device, graphics_family, 0, &graphics_queue);

    VkQueue present_queue;
    vkGetDeviceQueue(vk_device, present_family, 0, &present_queue);

    // Get surface info
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

    struct SurfaceInfo {
        VkSurfaceFormatKHR format;
        VkPresentModeKHR present_mode;
        VkSurfaceCapabilitiesKHR capabilities;
    };

    // TODO - Select optimal format, present mode and swap extent
    SurfaceInfo surface_info = {};
    surface_info.format = surface_formats[0];
    surface_info.present_mode = VK_PRESENT_MODE_FIFO_KHR;
    surface_info.capabilities = surface_capabilities;

    // Create render pass
    VkAttachmentDescription color_attachment = {};
    color_attachment.format = surface_info.format.format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo render_pass_create_info{};
    render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_create_info.attachmentCount = 1;
    render_pass_create_info.pAttachments = &color_attachment;
    render_pass_create_info.subpassCount = 1;
    render_pass_create_info.pSubpasses = &subpass;
    render_pass_create_info.dependencyCount = 1;
    render_pass_create_info.pDependencies = &dependency;

    VkRenderPass render_pass;
    VK_HANDLE_ERROR(
        vkCreateRenderPass(vk_device, &render_pass_create_info, nullptr, &render_pass),
        "Could not create render pass"
    );

    // Create shader
    auto read_file = [](const char* file_name) {
        std::ifstream fs(file_name, std::ios::ate | std::ios::binary);
        size_t file_size = fs.tellg();
        std::vector<char> file_buffer(file_size);
        fs.seekg(0);
        fs.read(file_buffer.data(), file_size);
        return file_buffer;
    };

    auto create_shader_module = [vk_device](std::vector<char>& shader_src) {
        VkShaderModuleCreateInfo shader_module_create_info = {};
        shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_module_create_info.codeSize = shader_src.size();
        shader_module_create_info.pCode = reinterpret_cast<const uint32_t*>(shader_src.data());

        VkShaderModule shader_module;
        VK_HANDLE_ERROR(
            vkCreateShaderModule(vk_device, &shader_module_create_info, nullptr, &shader_module),
            "Could not create shader module"
        );
        return shader_module;
    };

    auto vert_shader_src = read_file("vert.spv");
    auto frag_shader_src = read_file("frag.spv");
    auto vert_shader_module = create_shader_module(vert_shader_src);
    auto frag_shader_module = create_shader_module(frag_shader_src);

    VkPipelineShaderStageCreateInfo vert_shader_stage_create_info{};
    vert_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_shader_stage_create_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_shader_stage_create_info.module = vert_shader_module;
    vert_shader_stage_create_info.pName = "main";

    VkPipelineShaderStageCreateInfo frag_shader_stage_create_info{};
    frag_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_shader_stage_create_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_shader_stage_create_info.module = frag_shader_module;
    frag_shader_stage_create_info.pName = "main";

    VkPipelineShaderStageCreateInfo shader_stage_create_info[] = {
        vert_shader_stage_create_info, frag_shader_stage_create_info
    };

    // Create vertex buffers
    const float vertices[] = {
    //    X      Y     Z     R     G     B
         0.0f,  0.5f, 1.0f, 1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, 1.0f, 0.0f, 1.0f, 0.0f,
         0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 1.0f
    };

    VkVertexInputBindingDescription binding_description = {};
    binding_description.binding = 0;
    binding_description.stride = 6 * sizeof(float);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attribute_descriptions[2] = {};
    attribute_descriptions[0].binding = 0;
    attribute_descriptions[0].location = 0;
    attribute_descriptions[0].offset = 0;
    attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[1].binding = 0;
    attribute_descriptions[1].location = 1;
    attribute_descriptions[1].offset = 3 * sizeof(float);
    attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;

    VkBufferCreateInfo buffer_create_info = {};
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.size = sizeof(vertices);
    buffer_create_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer vertex_buffer;
    VK_HANDLE_ERROR(
        vkCreateBuffer(vk_device, &buffer_create_info, nullptr, &vertex_buffer),
        "Could not create vertex buffer"
    );

    // Memory
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(vk_device, vertex_buffer, &mem_requirements);

    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    uint32_t mem_type = -1;
    for(uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags properties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        if(
            (mem_requirements.memoryTypeBits & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties
        ) {
            mem_type = i;
            break;
        }
    }
    if(mem_type == -1) throw std::runtime_error("Could not find suitable memory type");

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = mem_type;

    VkDeviceMemory vertex_buffer_memory;
    VK_HANDLE_ERROR(
        vkAllocateMemory(vk_device, &alloc_info, nullptr, &vertex_buffer_memory),
        "Could not allocate vertex buffer memory"
    );
    vkBindBufferMemory(vk_device, vertex_buffer, vertex_buffer_memory, 0);

    void* buffer_data;
    vkMapMemory(vk_device, vertex_buffer_memory, 0, buffer_create_info.size, 0, &buffer_data);
    memcpy(buffer_data, vertices, buffer_create_info.size);
    vkUnmapMemory(vk_device, vertex_buffer_memory);

    // Pipeline
    std::vector<VkDynamicState> dynamic_states = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = dynamic_states.size();
    dynamic_state.pDynamicStates = dynamic_states.data();

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 1;
    vertex_input_info.pVertexBindingDescriptions = &binding_description;
    vertex_input_info.vertexAttributeDescriptionCount = 2;
    vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions;

    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = surface_info.capabilities.currentExtent.width;
    viewport.height = surface_info.capabilities.currentExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState color_blend_attachment = {};
    color_blend_attachment.colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT
        | VK_COLOR_COMPONENT_G_BIT
        | VK_COLOR_COMPONENT_B_BIT
        | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_TRUE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo color_blending = {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{};
    pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = 0;
    pipeline_layout_create_info.pushConstantRangeCount = 0;

    VkPipelineLayout pipeline_layout;
    VK_HANDLE_ERROR(
        vkCreatePipelineLayout(vk_device, &pipeline_layout_create_info, nullptr, &pipeline_layout),
        "Could not create pipeline layout"
    );

    VkGraphicsPipelineCreateInfo pipeline_create_info = {};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_create_info.stageCount = 2;
    pipeline_create_info.pStages = shader_stage_create_info;
    pipeline_create_info.pVertexInputState = &vertex_input_info;
    pipeline_create_info.pInputAssemblyState = &input_assembly;
    pipeline_create_info.pViewportState = &viewport_state;
    pipeline_create_info.pRasterizationState = &rasterizer;
    pipeline_create_info.pMultisampleState = &multisampling;
    pipeline_create_info.pDepthStencilState = nullptr;
    pipeline_create_info.pColorBlendState = &color_blending;
    pipeline_create_info.pDynamicState = &dynamic_state;
    pipeline_create_info.layout = pipeline_layout;
    pipeline_create_info.renderPass = render_pass;
    pipeline_create_info.subpass = 0;

    VkPipeline graphics_pipeline;
    VK_HANDLE_ERROR(
        vkCreateGraphicsPipelines(vk_device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &graphics_pipeline),
        "Could not create graphics pipeline"
    );

    vkDestroyShaderModule(vk_device, vert_shader_module, nullptr);
    vkDestroyShaderModule(vk_device, frag_shader_module, nullptr);


    // Create swapchain
    auto create_swapchain = [](
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
    };

    // Retrieve the swap chain images
    auto create_swapchain_image_views = [](VkDevice device, VkSwapchainKHR swapchain, VkFormat image_format) {
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
    };

    // Create framebuffers
    auto create_framebuffer = [](
        VkDevice device,
        VkRenderPass render_pass,
        VkImageView image_view,
        VkExtent2D extent
    ) {
        VkFramebufferCreateInfo framebuffer_create_info = {};
        framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_create_info.renderPass = render_pass;
        framebuffer_create_info.attachmentCount = 1;
        framebuffer_create_info.pAttachments = &image_view;
        framebuffer_create_info.width = extent.width;
        framebuffer_create_info.height = extent.height;
        framebuffer_create_info.layers = 1;

        VkFramebuffer framebuffer;
        VK_HANDLE_ERROR(
            vkCreateFramebuffer(device, &framebuffer_create_info, nullptr, &framebuffer),
            "Could not create framebuffer"
        );

        return framebuffer;
    };

    // Recreate swapchain
    auto recreate_swapchain = [create_swapchain, create_swapchain_image_views, create_framebuffer](
        VkDevice device,
        VkSwapchainKHR* swapchain,
        std::vector<VkImageView>* image_views,
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
            vkDestroySwapchainKHR(device, *swapchain, nullptr);
        }

        *swapchain = create_swapchain(device, surface, surface_info, graphics_family, present_family);
        *image_views = create_swapchain_image_views(device, *swapchain, surface_info.format.format);
        *framebuffers = std::vector<VkFramebuffer>(image_views->size());
        for(size_t i = 0; i < image_views->size(); ++i) {
            (*framebuffers)[i] = create_framebuffer(
                device, render_pass, (*image_views)[i], surface_info.capabilities.currentExtent
            );
        }
    };

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImageView> swapchain_image_views;
    std::vector<VkFramebuffer> framebuffers;
    recreate_swapchain(
        vk_device,
        &swapchain, &swapchain_image_views, &framebuffers,
        surface, surface_info,
        graphics_family, present_family,
        render_pass
    );

    // Create command pool
    VkCommandPoolCreateInfo command_pool_create_info = {};
    command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_create_info.queueFamilyIndex = graphics_family;
    command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool command_pool;
    VK_HANDLE_ERROR(
        vkCreateCommandPool(vk_device, &command_pool_create_info, nullptr, &command_pool),
        "Could not create command pool"
    );

    // Allocate command buffer
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> command_buffers;
    VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
    command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.commandPool = command_pool;
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    VK_HANDLE_ERROR(
        vkAllocateCommandBuffers(vk_device, &command_buffer_allocate_info, command_buffers.data()),
        "Could not allocate command buffer"
    );

    // Record command buffer
    auto record_cmd_buffer = [&](VkCommandBuffer command_buffer, uint32_t image_index) {
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        VK_HANDLE_ERROR(
            vkBeginCommandBuffer(command_buffer, &begin_info),
            "Could not begin command buffer"
        );

        VkClearValue clear_color = { 0.0f, 0.0f, 0.0f, 1.0f };
        VkRenderPassBeginInfo render_pass_begin_info = {};
        render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_begin_info.renderPass = render_pass;
        render_pass_begin_info.framebuffer = framebuffers[image_index];
        render_pass_begin_info.renderArea.offset = {0, 0};
        render_pass_begin_info.renderArea.extent = surface_info.capabilities.currentExtent;
        render_pass_begin_info.clearValueCount = 1;
        render_pass_begin_info.pClearValues = &clear_color;

        vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, offsets);

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = surface_info.capabilities.currentExtent;
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        vkCmdDraw(command_buffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(command_buffer);
        VK_HANDLE_ERROR(
            vkEndCommandBuffer(command_buffer),
            "Could not end command buffer"
        );
    };

    // Create synchronization primitives
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> image_avaiable;
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> render_finished;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> frame_in_flight;
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkSemaphoreCreateInfo semaphore_create_info = {};
        semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VK_HANDLE_ERROR(
            vkCreateSemaphore(vk_device, &semaphore_create_info, nullptr, &image_avaiable[i]),
            "Could not create semaphore"
        );
        VK_HANDLE_ERROR(
            vkCreateSemaphore(vk_device, &semaphore_create_info, nullptr, &render_finished[i]),
            "Could not create semaphore"
        );

        VkFenceCreateInfo fence_create_info = {};
        fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        VK_HANDLE_ERROR(
            vkCreateFence(vk_device, &fence_create_info, nullptr, &frame_in_flight[i]),
            "Could not create fence"
        );
    }

    //

    uint32_t current_frame = 0;
    bool framebuffer_resized = false;

    // Run application
    SDL_Event e; bool quit = false;
    while(quit == false) {
        while(SDL_PollEvent(&e)) {
            if(e.type == SDL_QUIT) quit = true;
            if(e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED) framebuffer_resized = true;
        }

        // Render frame
        vkWaitForFences(vk_device, 1, &frame_in_flight[current_frame], VK_TRUE, UINT64_MAX);

        uint32_t image_index;
        VkResult r = vkAcquireNextImageKHR(vk_device, swapchain, UINT64_MAX, image_avaiable[current_frame], VK_NULL_HANDLE, &image_index);
        if(r == VK_ERROR_OUT_OF_DATE_KHR) {
            vkDeviceWaitIdle(vk_device);
            recreate_swapchain(
                vk_device,
                &swapchain, &swapchain_image_views, &framebuffers,
                surface, surface_info,
                graphics_family, present_family,
                render_pass
            );
            continue;
        }

        vkResetFences(vk_device, 1, &frame_in_flight[current_frame]);

        vkResetCommandBuffer(command_buffers[current_frame], 0);
        record_cmd_buffer(command_buffers[current_frame], image_index);

        VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &image_avaiable[current_frame];
        submit_info.pWaitDstStageMask = wait_stages;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffers[current_frame];
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_finished[current_frame];

        VK_HANDLE_ERROR(
            vkQueueSubmit(graphics_queue, 1, &submit_info, frame_in_flight[current_frame]),
            "Could not submit queue"
        );

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished[current_frame];
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain;
        present_info.pImageIndices = &image_index;

        r = vkQueuePresentKHR(present_queue, &present_info);
        if(r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || framebuffer_resized) {
            framebuffer_resized = false;
            vkDeviceWaitIdle(vk_device);
            recreate_swapchain(
                vk_device,
                &swapchain, &swapchain_image_views, &framebuffers,
                surface, surface_info,
                graphics_family, present_family,
                render_pass
            );
            continue;
        }

        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    vkDeviceWaitIdle(vk_device);

    // Clean up
    vkDestroyBuffer(vk_device, vertex_buffer, nullptr);
    vkFreeMemory(vk_device, vertex_buffer_memory, nullptr);
    for(auto f : frame_in_flight) vkDestroyFence(vk_device, f, nullptr);
    for(auto s : render_finished) vkDestroySemaphore(vk_device, s, nullptr);
    for(auto s : image_avaiable) vkDestroySemaphore(vk_device, s, nullptr);
    vkDestroyCommandPool(vk_device, command_pool, nullptr);
    for(const auto framebuffer : framebuffers) {
        vkDestroyFramebuffer(vk_device, framebuffer, nullptr);
    }
    vkDestroyPipeline(vk_device, graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(vk_device, pipeline_layout, nullptr);
    vkDestroyRenderPass(vk_device, render_pass, nullptr);
    for(const auto image_view : swapchain_image_views) {
        vkDestroyImageView(vk_device, image_view, nullptr);
    }
    vkDestroySwapchainKHR(vk_device, swapchain, nullptr);
    vkDestroyDevice(vk_device, nullptr);
    vkDestroySurfaceKHR(vk_instance, surface, nullptr);
    vkDestroyInstance(vk_instance, nullptr);

    SDL_DestroyWindow(window);
    SDL_Quit();
}
