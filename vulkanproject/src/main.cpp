#include "SDL_keyboard.h"
#include "SDL_mouse.h"
#include "labhelper.hpp"
#include "model.hpp"
#include <optional>
#include <vulkan/vulkan_core.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <vector>
#include <array>
#include <imgui.h>
#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_vulkan.h>
#include <chrono>

const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

int main() {
    // ------------------------------------------------------------
    // Project setup
    // ------------------------------------------------------------

    // Init
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Vulkan_LoadLibrary(nullptr);

    // Create window
    SDL_Window* window = SDL_CreateWindow(
        "Vulkan project",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        SCREEN_WIDTH, SCREEN_HEIGHT,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN
    );

    // Init vulkan
    VkInstance vk_instance;
    VkDevice vk_device;
    VkPhysicalDevice physical_device;
    uint32_t graphics_family, present_family;
    VkQueue graphics_queue, present_queue;
    VkCommandPool command_pool;
    VkSurfaceKHR surface;
    init_vulkan(
        window,
        &vk_instance,
        &vk_device, &physical_device,
        &graphics_family, &present_family,
        &graphics_queue, &present_queue,
        &command_pool,
        &surface
    );

    SurfaceInfo surface_info = get_surface_info(physical_device, surface);

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

    VkAttachmentDescription depth_attachment = {};
    depth_attachment.format = VK_FORMAT_D32_SFLOAT;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkRenderPass render_pass = create_render_pass(
        vk_device, 
        color_attachment, depth_attachment
    );

    // Create swapchain
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImageView> swapchain_image_views;
    VkImage depth_image;
    VkDeviceMemory depth_image_memory;
    VkImageView depth_image_view;
    std::vector<VkFramebuffer> framebuffers;
    recreate_swapchain(
        vk_device, physical_device,
        command_pool, graphics_queue,
        &swapchain, &swapchain_image_views,
        &depth_image, &depth_image_memory, &depth_image_view,
        &framebuffers,
        surface, surface_info,
        graphics_family, present_family,
        render_pass
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

    // ImGUI
    VkDescriptorPool imgui_descriptor_pool = imgui_init(
        vk_instance,
        window,
        vk_device, physical_device,
        graphics_family, graphics_queue,
        render_pass
    );

    // ------------------------------------------------------------
    // Get misc resources
    // ------------------------------------------------------------

    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);

    VkSampler sampler;
    VkSamplerCreateInfo sampler_create_info = {};
    sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_create_info.minLod = 0.0f;
    sampler_create_info.maxLod = VK_LOD_CLAMP_NONE;
    sampler_create_info.compareEnable = VK_FALSE;
    sampler_create_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_create_info.minFilter = VK_FILTER_LINEAR;
    sampler_create_info.magFilter = VK_FILTER_LINEAR;
    sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_create_info.mipLodBias = 0.0f;
    sampler_create_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.anisotropyEnable = VK_TRUE;
    sampler_create_info.maxAnisotropy = device_properties.limits.maxSamplerAnisotropy;
    sampler_create_info.unnormalizedCoordinates = VK_FALSE;
    vkCreateSampler(vk_device, &sampler_create_info, nullptr, &sampler);

    // Load environment map
    Texture env_map = load_texture_from_image(
        vk_device, physical_device,
        command_pool, graphics_queue,
        "scenes/envmaps/001.hdr"
    );
    Texture irradiance_map = load_texture_from_image(
        vk_device, physical_device,
        command_pool, graphics_queue,
        "scenes/envmaps/001_irradiance.hdr"
    );
    std::vector<const char*> reflection_lods = {
        "scenes/envmaps/001_dl_0.hdr",
        "scenes/envmaps/001_dl_1.hdr",
        "scenes/envmaps/001_dl_2.hdr",
        "scenes/envmaps/001_dl_3.hdr",
        "scenes/envmaps/001_dl_4.hdr",
        "scenes/envmaps/001_dl_5.hdr",
        "scenes/envmaps/001_dl_6.hdr",
        "scenes/envmaps/001_dl_7.hdr",
    };

    Texture reflection_map = load_texture_from_image(
        vk_device, physical_device,
        command_pool, graphics_queue,
        reflection_lods[0],
        reflection_lods.size()
    );
    transition_image_layout(
        vk_device,
        command_pool, graphics_queue,
        reflection_map.m_image,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        reflection_lods.size()
    );
    for(size_t i = 1; i < reflection_lods.size(); ++i) {
        int w, h, c;
        stbi_uc* pixels = stbi_load(
            reflection_lods[i],
            &w, &h, &c,
            4
        );
        if(pixels == nullptr) throw std::runtime_error("Could not load image");

        size_t mip_size = w * h * 4;

        write_image_staged(
            vk_device, physical_device,
            graphics_queue, command_pool,
            reflection_map.m_image,
            pixels, mip_size,
            w, h,
            i
        );

        stbi_image_free(pixels);
    }
    transition_image_layout(
        vk_device,
        command_pool, graphics_queue,
        reflection_map.m_image,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        reflection_lods.size()
    );

    // Create quad
    VkVertexInputBindingDescription quad_binding_description = {};
    quad_binding_description.binding = 0;
    quad_binding_description.stride = 2 * sizeof(float);
    quad_binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription quad_position_attribute;
    quad_position_attribute.binding = 0;
    quad_position_attribute.location = 0;
    quad_position_attribute.offset = 0;
    quad_position_attribute.format = VK_FORMAT_R32G32_SFLOAT;

    std::array<float, 6 * 2> quad_vertices = {
        -1.0,  1.0,   1.0,  1.0,   1.0, -1.0,
         1.0, -1.0,  -1.0, -1.0,  -1.0,  1.0
    };

    VkBuffer quad_vertex_buffer = create_buffer(
        vk_device,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        quad_vertices.size() * sizeof(float)
    );
    VkDeviceMemory quad_vertex_memory = allocate_buffer_memory(
        vk_device, physical_device,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        quad_vertex_buffer
    );
    write_buffer_staged(
        vk_device, physical_device,
        graphics_queue, command_pool,
        quad_vertex_buffer,
        quad_vertices.data(), quad_vertices.size() * sizeof(float)
    );

    VkDescriptorSetLayout material_descriptor_set_layout =
        create_model_descriptor_set_layout(vk_device);


    // ------------------------------------------------------------
    // Load scene
    // ------------------------------------------------------------

    // Load objects
    Model spaceship_model = load_model_from_file(
        vk_device, physical_device, command_pool, graphics_queue,
        "scenes/space-ship.obj"
    );
    Model materialtest_model = load_model_from_file(
        vk_device, physical_device, command_pool, graphics_queue,
        "scenes/materialtest.obj"
    );
    Model sphere_model = load_model_from_file(
        vk_device, physical_device, command_pool, graphics_queue,
        "scenes/sphere.obj"
    );
    Model landingpad_model = load_model_from_file(
        vk_device, physical_device, command_pool, graphics_queue,
        "scenes/landingpad.obj"
    );
    std::vector<Model*> models = {
        &spaceship_model,
        &materialtest_model,
        &sphere_model,
        &landingpad_model,
    };

    Object spaceship_object = {};
    spaceship_object.position = glm::vec3(0.0, 8.0, 0.0);
    spaceship_object.orientation = glm::identity<glm::quat>();
    spaceship_object.scale = glm::one<glm::vec3>();
    spaceship_object.m_model_index = 0;

    Object materialtest_object = {};
    materialtest_object.position = glm::vec3(0.0, 0.0, 0.0);
    materialtest_object.orientation = glm::identity<glm::quat>();
    materialtest_object.scale = glm::one<glm::vec3>();
    materialtest_object.m_model_index = 1;

    Object light_object = {};
    light_object.position = glm::vec3(20.0, 20.0, 20.0);
    light_object.orientation = glm::identity<glm::quat>();
    light_object.scale = 0.1f * glm::one<glm::vec3>();
    light_object.m_model_index = 2;

    Object landingpad_object = {};
    landingpad_object.position = glm::vec3(0.0, 0.0, 0.0);
    landingpad_object.orientation = glm::identity<glm::quat>();
    landingpad_object.scale = glm::one<glm::vec3>();
    landingpad_object.m_model_index = 3;

    std::vector<Object*> objects = {
        &spaceship_object,
        &light_object,
        &landingpad_object
    };

    // Create uniform buffers
    glm::mat4 view_matrix = glm::identity<glm::mat4>();
    glm::mat4 projection_matrix = glm::perspective(glm::radians(45.0), (640.0 / 480.0), 0.01, 400.0);
    projection_matrix[1][1] *= -1.0;

    glm::vec3 view_space_light_pos = view_matrix * glm::vec4(light_object.position, 1.0);

    FrameData frame_data;
    frame_data = create_frame_data(
        vk_device, physical_device, command_pool,
        graphics_queue,
        material_descriptor_set_layout,
        100
    );

    update_frame_data(
        vk_device,
        &frame_data,
        sampler,
        objects,
        models,
        view_matrix,
        projection_matrix
    );

    // ------------------------------------------------------------
    // Shading graphics pipeline
    // ------------------------------------------------------------

    // Specify descriptors
    struct GlobalUBO {
        glm::mat4 light_matrix;
        glm::mat4 view_inverse;
        glm::vec3 view_space_light_position;
        float light_intensity;
        glm::vec3 light_color;
        float env_multiplier;
        glm::vec3 light_view_dir;
    };

    VkBuffer global_ubo_buffer = create_buffer(
        vk_device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GlobalUBO)
    );
    VkDeviceMemory global_ubo_memory = allocate_buffer_memory(
        vk_device, physical_device,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        global_ubo_buffer
    );

    std::vector<DescriptorInfo> global_descriptors = {
        {
            0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT,
            std::make_optional(global_ubo_buffer), std::make_optional(sizeof(GlobalUBO)),
            std::nullopt, std::nullopt
        },
        {
            1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::nullopt, std::nullopt,
            std::make_optional(env_map.m_image_view), std::make_optional(sampler)
        },
        {
            2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::nullopt, std::nullopt,
            std::make_optional(irradiance_map.m_image_view), std::make_optional(sampler)
        },
        {
            3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::nullopt, std::nullopt,
            std::make_optional(reflection_map.m_image_view), std::make_optional(sampler)
        },
        /* { */
        /*     4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, */
        /*     std::nullopt, std::nullopt, */
        /*     std::make_optional(shadowmap_depth_textures[0].m_image_view), std::make_optional(shadow_sampler) */
        /* } */
    };

    VkDescriptorPool global_descriptor_pool;
    VkDescriptorSetLayout global_descriptor_set_layout;
    VkDescriptorSet global_descriptor_set;
    create_descriptors(
        vk_device,
        &global_descriptor_pool,
        &global_descriptor_set_layout, &global_descriptor_set,
        global_descriptors
    );

    update_descriptors(vk_device, global_descriptor_set, global_descriptors);

    // Specify vertex data description
    auto model_attributes = create_model_attributes();

    // Shader
    std::vector<char> material_vert_shader_src = read_file("vulkanproject/vert.spv");
    std::vector<char> material_frag_shader_src = read_file("vulkanproject/frag.spv");
    VkShaderModule material_vert_shader_module = create_shader_module(
        vk_device, material_vert_shader_src
    );
    VkShaderModule material_frag_shader_module = create_shader_module(
        vk_device, material_frag_shader_src
    );

    // Pipeline
    VkPipelineLayout material_pipeline_layout = create_pipeline_layout(
        vk_device,
        { material_descriptor_set_layout, global_descriptor_set_layout }
    );

    VkPipeline material_pipeline = create_graphics_pipeline(
        vk_device,
        material_pipeline_layout,
        render_pass,
        { model_attributes.first },
        model_attributes.second,
        material_vert_shader_module,
        material_frag_shader_module
    );

    vkDestroyShaderModule(vk_device, material_vert_shader_module, nullptr);
    vkDestroyShaderModule(vk_device, material_frag_shader_module, nullptr);

    // ------------------------------------------------------------
    // Background render pipeline
    // ------------------------------------------------------------

    // Shader
    std::vector<char> bg_vert_shader_src = read_file("vulkanproject/bg_vert.spv");
    std::vector<char> bg_frag_shader_src = read_file("vulkanproject/bg_frag.spv");
    VkShaderModule bg_vert_shader_module = create_shader_module(
        vk_device, bg_vert_shader_src
    );
    VkShaderModule bg_frag_shader_module = create_shader_module(
        vk_device, bg_frag_shader_src
    );

    // Descriptors
    struct BgUniformBlock {
        glm::mat4 inv_pv;
        glm::vec3 camera_pos;
        float environment_multiplier;
    };

    VkBuffer bg_ubo_buffer = create_buffer(
        vk_device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(BgUniformBlock)
    );
    VkDeviceMemory bg_ubo_memory = allocate_buffer_memory(
        vk_device, physical_device,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        bg_ubo_buffer
    );
    std::vector<DescriptorInfo> bg_descriptors = {
        {
            0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::make_optional(bg_ubo_buffer), std::make_optional(sizeof(BgUniformBlock)),
            std::nullopt, std::nullopt
        },
        {
            1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::nullopt, std::nullopt,
            std::make_optional(env_map.m_image_view), std::make_optional(sampler)
        }
    };

    VkDescriptorPool bg_descriptor_pool;
    VkDescriptorSetLayout bg_descriptor_set_layout;
    VkDescriptorSet bg_descriptor_set;
    create_descriptors(
        vk_device,
        &bg_descriptor_pool,
        &bg_descriptor_set_layout, &bg_descriptor_set,
        bg_descriptors
    );

    update_descriptors(vk_device, bg_descriptor_set, bg_descriptors);

    // Pipeline
    VkPipelineLayout bg_pipeline_layout = create_pipeline_layout(
        vk_device,
        { bg_descriptor_set_layout }
    );

    VkPipelineDepthStencilStateCreateInfo no_depth_stencil = {};
    no_depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    no_depth_stencil.depthTestEnable = VK_FALSE;
    no_depth_stencil.depthWriteEnable = VK_FALSE;
    no_depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    no_depth_stencil.depthBoundsTestEnable = VK_FALSE;
    no_depth_stencil.minDepthBounds = 0.0f;
    no_depth_stencil.maxDepthBounds = 1.0f;
    no_depth_stencil.stencilTestEnable = VK_FALSE;
    no_depth_stencil.front = {};
    no_depth_stencil.back = {};

    VkPipeline bg_pipeline = create_graphics_pipeline(
        vk_device,
        bg_pipeline_layout,
        render_pass,
        { quad_binding_description },
        { quad_position_attribute },
        bg_vert_shader_module,
        bg_frag_shader_module,
        std::nullopt,
        no_depth_stencil
    );

    vkDestroyShaderModule(vk_device, bg_vert_shader_module, nullptr);
    vkDestroyShaderModule(vk_device, bg_frag_shader_module, nullptr);

    // ------------------------------------------------------------
    // G-buffer pipeline
    // ------------------------------------------------------------

    // Create render pass
    VkAttachmentDescription normal_attachment = color_attachment;
    normal_attachment.format = VK_FORMAT_R16G16B16A16_SNORM;

    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderPass gbuffer_render_pass = create_render_pass(
        vk_device, 
        normal_attachment, depth_attachment
    );

    // G-buffer framebuffers
    VkFramebuffer gbuffer_framebuffer;
    Texture gbuffer_normals_texture;
    Texture gbuffer_depth_texture;
    create_framebuffer_complete(
        vk_device, physical_device,
        command_pool, graphics_queue,
        gbuffer_render_pass, surface_info.capabilities.currentExtent, VK_IMAGE_USAGE_SAMPLED_BIT,
        &gbuffer_framebuffer, &gbuffer_normals_texture, &gbuffer_depth_texture,
        normal_attachment.format
    );

    std::vector<char> gbuffer_vert_shader_src = read_file("vulkanproject/gbuffer_vert.spv");
    std::vector<char> gbuffer_frag_shader_src = read_file("vulkanproject/gbuffer_frag.spv");
    VkShaderModule gbuffer_vert_shader_module = create_shader_module(
        vk_device, gbuffer_vert_shader_src
    );
    VkShaderModule gbuffer_frag_shader_module = create_shader_module(
        vk_device, gbuffer_frag_shader_src
    );

    // Pipeline
    VkPipelineLayout gbuffer_pipeline_layout = create_pipeline_layout(
        vk_device,
        { material_descriptor_set_layout }
    );

    VkPipeline gbuffer_pipeline = create_graphics_pipeline(
        vk_device,
        gbuffer_pipeline_layout,
        gbuffer_render_pass,
        { model_attributes.first },
        model_attributes.second,
        gbuffer_vert_shader_module,
        gbuffer_frag_shader_module
    );

    vkDestroyShaderModule(vk_device, gbuffer_vert_shader_module, nullptr);
    vkDestroyShaderModule(vk_device, gbuffer_frag_shader_module, nullptr);

    // ------------------------------------------------------------
    // SSAO pipeline
    // ------------------------------------------------------------

    // Shader
    std::vector<char> ssao_vert_shader_src = read_file("vulkanproject/ssao_vert.spv");
    std::vector<char> ssao_frag_shader_src = read_file("vulkanproject/ssao_frag.spv");
    VkShaderModule ssao_vert_shader_module = create_shader_module(
        vk_device, ssao_vert_shader_src
    );
    VkShaderModule ssao_frag_shader_module = create_shader_module(
        vk_device, ssao_frag_shader_src
    );

    // Descriptors
    struct SSAOUniformBlock {
        glm::mat4 inv_pv;
        glm::mat4 projection_matrix;
        glm::mat4 inv_projection_matrix;
        std::array<glm::vec4, 128> samples;
    };

    VkBuffer ssao_ubo_buffer = create_buffer(
        vk_device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(SSAOUniformBlock)
    );
    VkDeviceMemory ssao_ubo_memory = allocate_buffer_memory(
        vk_device, physical_device,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        ssao_ubo_buffer
    );
    std::vector<DescriptorInfo> ssao_descriptors = {
        {
            0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::make_optional(ssao_ubo_buffer), std::make_optional(sizeof(SSAOUniformBlock)),
            std::nullopt, std::nullopt
        },
        {
            1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::nullopt, std::nullopt,
            std::make_optional(gbuffer_normals_texture.m_image_view), std::make_optional(sampler)
        },
        {
            2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::nullopt, std::nullopt,
            std::make_optional(gbuffer_depth_texture.m_image_view), std::make_optional(sampler)
        }
    };

    VkDescriptorPool ssao_descriptor_pool;
    VkDescriptorSetLayout ssao_descriptor_set_layout;
    VkDescriptorSet ssao_descriptor_set;
    create_descriptors(
        vk_device,
        &ssao_descriptor_pool,
        &ssao_descriptor_set_layout, &ssao_descriptor_set,
        ssao_descriptors
    );

    update_descriptors(vk_device, ssao_descriptor_set, ssao_descriptors);

    // Pipeline
    VkPipelineLayout ssao_pipeline_layout = create_pipeline_layout(
        vk_device,
        { ssao_descriptor_set_layout }
    );

    VkPipeline ssao_pipeline = create_graphics_pipeline(
        vk_device,
        ssao_pipeline_layout,
        render_pass,
        { quad_binding_description },
        { quad_position_attribute },
        ssao_vert_shader_module,
        ssao_frag_shader_module,
        std::nullopt,
        no_depth_stencil
    );

    vkDestroyShaderModule(vk_device, ssao_vert_shader_module, nullptr);
    vkDestroyShaderModule(vk_device, ssao_frag_shader_module, nullptr);

    // ------------------------------------------------------------
    // Shadowmap render pipeline
    // ------------------------------------------------------------
    /* // Shadow map framebuffers */
    /* std::array<VkFramebuffer, MAX_FRAMES_IN_FLIGHT> shadowmap_framebuffers; */
    /* std::array<Texture, MAX_FRAMES_IN_FLIGHT> shadowmap_color_textures; */
    /* std::array<Texture, MAX_FRAMES_IN_FLIGHT> shadowmap_depth_textures; */
    /* for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) { */
    /*     create_framebuffer_complete( */
    /*         vk_device, physical_device, */
    /*         command_pool, graphics_queue, */
    /*         render_pass, {1024, 1024}, VK_IMAGE_USAGE_SAMPLED_BIT, */
    /*         &shadowmap_framebuffers[i], &shadowmap_color_textures[i], &shadowmap_depth_textures[i] */
    /*     ); */
    /* } */

    /* // Shader */
    /* std::vector<char> shadowmap_vert_shader_src = read_file("vulkanproject/shadowmap_vert.spv"); */
    /* std::vector<char> shadowmap_frag_shader_src = read_file("vulkanproject/shadowmap_frag.spv"); */
    /* VkShaderModule shadowmap_vert_shader_module = create_shader_module( */
    /*     vk_device, shadowmap_vert_shader_src */
    /* ); */
    /* VkShaderModule shadowmap_frag_shader_module = create_shader_module( */
    /*     vk_device, shadowmap_frag_shader_src */
    /* ); */

    /* // Pipeline */
    /* VkPipelineLayout shadowmap_pipeline_layout = create_pipeline_layout( */
    /*     vk_device, */
    /*     { material_descriptor_set_layout } */
    /* ); */

    /* VkPipelineRasterizationStateCreateInfo shadowmap_rasterizer = {}; */
    /* shadowmap_rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO; */
    /* shadowmap_rasterizer.depthClampEnable = VK_FALSE; */
    /* shadowmap_rasterizer.rasterizerDiscardEnable = VK_FALSE; */
    /* shadowmap_rasterizer.polygonMode = VK_POLYGON_MODE_FILL; */
    /* shadowmap_rasterizer.lineWidth = 1.0f; */
    /* shadowmap_rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; */
    /* shadowmap_rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; */
    /* shadowmap_rasterizer.depthBiasEnable = VK_TRUE; */
    /* shadowmap_rasterizer.depthBiasSlopeFactor = 2.5; */

    /* VkPipeline shadowmap_pipeline = create_graphics_pipeline( */
    /*     vk_device, */
    /*     shadowmap_pipeline_layout, */
    /*     render_pass, */
    /*     { model_attributes.first }, */
    /*     model_attributes.second, */
    /*     shadowmap_vert_shader_module, */
    /*     shadowmap_frag_shader_module, */
    /*     shadowmap_rasterizer */
    /* ); */

    //

    uint32_t current_frame = 0;
    bool framebuffer_resized = false;
    glm::vec3 camera_position = -glm::vec3(-30.0, 10.0, 30.0);
    glm::vec3 camera_forward = glm::normalize(-glm::vec3(-30.0, 5.0, 30.0));

    float light_distance = 55.0;
    float light_azimuth = 0.0;
    float light_zenith = 45.0;
    glm::mat3 light_rot =
        glm::rotate(glm::radians(light_azimuth), glm::vec3(0, 1, 0))
        * glm::rotate(glm::radians(light_zenith), glm::vec3(0, 0, 1));

    auto start_time = std::chrono::high_resolution_clock::now();
    float previous_frame_time = 0.0f;
    bool regenerate_shadowmap_pipeline = false;

    BgUniformBlock bg_ubo = {};
    bg_ubo.inv_pv = glm::inverse(projection_matrix * view_matrix);
    bg_ubo.camera_pos = camera_position;
    bg_ubo.environment_multiplier = 0.5;
    write_memory_mapped(vk_device, bg_ubo_memory, bg_ubo);

    GlobalUBO global_ubo = {};
    global_ubo.light_matrix = glm::identity<glm::mat4>();
    global_ubo.view_inverse = glm::inverse(view_matrix);
    global_ubo.view_space_light_position = view_matrix * glm::vec4(light_object.position, 1.0);
    global_ubo.light_color = glm::vec3(1.0, 1.0, 1.0);
    global_ubo.light_intensity = 800.0;
    global_ubo.env_multiplier = 0.1;
    global_ubo.light_view_dir = glm::normalize(
        glm::vec3(view_matrix * glm::vec4(-light_object.position, 0.0))
    );

    SSAOUniformBlock ssao_ubo = {};
    ssao_ubo.inv_pv = glm::inverse(projection_matrix * view_matrix);
    ssao_ubo.projection_matrix = projection_matrix;
    ssao_ubo.inv_projection_matrix = glm::inverse(projection_matrix);
    for(size_t i = 0; i < ssao_ubo.samples.size(); ++i) {
        ssao_ubo.samples[i] = glm::vec4(cosine_sample_hemisphere(), 0.0);
    }

    // ------------------------------------------------------------
    // Render passes
    // ------------------------------------------------------------

    // Record command buffer for frame rendering
    glm::vec4 clear_color(0.0, 0.0, 0.0, 1.0);
    /* auto render_shadowmap = [&]( */
    /*     VkCommandBuffer command_buffer, */
    /*     uint32_t image_index, uint32_t current_frame, */
    /*     glm::mat4 light_view_matrix, glm::mat4 light_projection_matrix */
    /* ) { */
    /*     update_frame_data( */
    /*         vk_device, */
    /*         &frame_data[current_frame], */
    /*         sampler, objects, models, */
    /*         light_view_matrix, light_projection_matrix */
    /*     ); */
    /*     VkCommandBufferBeginInfo begin_info = {}; */
    /*     begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; */
    /*     VK_HANDLE_ERROR( */
    /*         vkBeginCommandBuffer(command_buffer, &begin_info), */
    /*         "Could not begin command buffer" */
    /*     ); */

    /*     std::array<VkClearValue, 2> clear_values = {}; */
    /*     clear_values[0].color = { clear_color.r, clear_color.g, clear_color.b, clear_color.a }; */
    /*     clear_values[1].depthStencil = { 1.0f, 0 }; */
    /*     VkExtent2D render_extent = { */
    /*         static_cast<uint32_t>(shadowmap_color_textures[current_frame].m_width), */
    /*         static_cast<uint32_t>(shadowmap_color_textures[current_frame].m_height) */
    /*     }; */ 
    /*     VkViewport viewport = {}; */
    /*     viewport.x = 0.0f; */
    /*     viewport.y = 0.0f; */
    /*     viewport.width = render_extent.width; */
    /*     viewport.height = render_extent.height; */
    /*     viewport.minDepth = 0.0f; */
    /*     viewport.maxDepth = 1.0f; */
    /*     VkRect2D scissor = {}; */
    /*     scissor.offset = { 0, 0 }; */
    /*     scissor.extent = render_extent; */
    /*     VkDeviceSize offsets[] = { 0 }; */

    /*     // Security camera render pass */
    /*     VkRenderPassBeginInfo render_pass_begin_info = {}; */
    /*     render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO; */
    /*     render_pass_begin_info.renderPass = render_pass; */
    /*     render_pass_begin_info.framebuffer = shadowmap_framebuffers[current_frame]; */
    /*     render_pass_begin_info.renderArea.offset = {0, 0}; */
    /*     render_pass_begin_info.renderArea.extent = render_extent; */
    /*     render_pass_begin_info.clearValueCount = clear_values.size(); */
    /*     render_pass_begin_info.pClearValues = clear_values.data(); */
    /*     vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE); */
    /*     vkCmdSetViewport(command_buffer, 0, 1, &viewport); */
    /*     vkCmdSetScissor(command_buffer, 0, 1, &scissor); */

    /*     // Render scene */
    /*     vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowmap_pipeline); */

    /*     size_t descriptor_index = 0; */
    /*     for(const Object* object : objects) { */
    /*         const Model& model = *models[object->m_model_index]; */
    /*         vkCmdBindVertexBuffers(command_buffer, 0, 1, &model.m_vertex_buffer, offsets); */

    /*         for(const Mesh& mesh : model.m_meshes) { */
    /*             vkCmdBindDescriptorSets( */
    /*                 command_buffer, */
    /*                 VK_PIPELINE_BIND_POINT_GRAPHICS, */
    /*                 material_pipeline_layout, */
    /*                 0, 1, */
    /*                 &frame_data[current_frame].m_descriptor_sets[descriptor_index], */
    /*                 0, nullptr */
    /*             ); */
    /*             vkCmdDraw(command_buffer, mesh.m_num_vertices, 1, mesh.m_start_index, 0); */
                
    /*             descriptor_index++; */
    /*         } */
    /*     } */

    /*     vkCmdEndRenderPass(command_buffer); */
    /*     VK_HANDLE_ERROR( */
    /*         vkEndCommandBuffer(command_buffer), */
    /*         "Could not end command buffer" */
    /*     ); */
    /* }; */
    auto render_gbuffer = [&](VkCommandBuffer command_buffer, uint32_t image_index, uint32_t current_frame) {
        begin_render_pass(
            command_buffer, gbuffer_render_pass, gbuffer_framebuffer,
            surface_info.capabilities.currentExtent
        );

        VkDeviceSize offsets[] = { 0 };
        // Render scene
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, gbuffer_pipeline);

        size_t descriptor_index = 0;
        for(const Object* object : objects) {
            const Model& model = *models[object->m_model_index];
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &model.m_vertex_buffer, offsets);

            for(const Mesh& mesh : model.m_meshes) {
                vkCmdBindDescriptorSets(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    material_pipeline_layout,
                    0, 1,
                    &frame_data.m_descriptor_sets[descriptor_index],
                    0, nullptr
                );
                vkCmdDraw(command_buffer, mesh.m_num_vertices, 1, mesh.m_start_index, 0);

                descriptor_index++;
            }
        }

        vkCmdEndRenderPass(command_buffer);
        VK_HANDLE_ERROR(
            vkEndCommandBuffer(command_buffer),
            "Could not end command buffer"
        );

    };

    auto render_ssao = [&](VkCommandBuffer command_buffer, uint32_t image_index, uint32_t current_frame) {
        begin_render_pass(
            command_buffer, render_pass, framebuffers[image_index],
            surface_info.capabilities.currentExtent
        );

        VkDeviceSize offsets[] = { 0 };
        // Render scene
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ssao_pipeline);

        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            ssao_pipeline_layout,
            0, 1,
            &ssao_descriptor_set,
            0, nullptr
        );
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &quad_vertex_buffer, offsets);
        vkCmdDraw(command_buffer, quad_vertices.size(), 1, 0, 0);

        vkCmdEndRenderPass(command_buffer);
        VK_HANDLE_ERROR(
            vkEndCommandBuffer(command_buffer),
            "Could not end command buffer"
        );

    };

    auto render_frame = [&](VkCommandBuffer command_buffer, uint32_t image_index, uint32_t current_frame) {
        begin_render_pass(command_buffer, render_pass, framebuffers[image_index], surface_info.capabilities.currentExtent);

        VkDeviceSize offsets[] = { 0 };
        // Render background
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bg_pipeline);

        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            bg_pipeline_layout,
            0, 1,
            &bg_descriptor_set,
            0, nullptr
        );
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &quad_vertex_buffer, offsets);
        vkCmdDraw(command_buffer, quad_vertices.size(), 1, 0, 0);

        // Render scene
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material_pipeline);

        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            material_pipeline_layout,
            1, 1,
            &global_descriptor_set,
            0, nullptr
        );

        size_t descriptor_index = 0;
        for(const Object* object : objects) {
            const Model& model = *models[object->m_model_index];
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &model.m_vertex_buffer, offsets);

            for(const Mesh& mesh : model.m_meshes) {
                vkCmdBindDescriptorSets(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    material_pipeline_layout,
                    0, 1,
                    &frame_data.m_descriptor_sets[descriptor_index],
                    0, nullptr
                );
                vkCmdDraw(command_buffer, mesh.m_num_vertices, 1, mesh.m_start_index, 0);

                descriptor_index++;
            }
        }

        // Render GUI
        imgui_new_frame();

        /* float depth_bias_offset = shadowmap_rasterizer.depthBiasSlopeFactor; */
        /* ImGui::SliderFloat("Depth bias offset", &shadowmap_rasterizer.depthBiasSlopeFactor, 0.0, 10.0); */
        /* if(shadowmap_rasterizer.depthBiasSlopeFactor != depth_bias_offset) { */
        /*     regenerate_shadowmap_pipeline = true; */
        /* } */

        imgui_render(command_buffer);

        vkCmdEndRenderPass(command_buffer);
        VK_HANDLE_ERROR(
            vkEndCommandBuffer(command_buffer),
            "Could not end command buffer"
        );

    };

    // ------------------------------------------------------------
    // Main program loop
    // ------------------------------------------------------------

    SDL_Event e; bool quit = false;
    while(quit == false) {
        while(SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if(e.type == SDL_QUIT) quit = true;
            if(e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED) framebuffer_resized = true;
        }

        // ------------------------------------------------------------
        // Program logic
        // ------------------------------------------------------------
		std::chrono::duration<float> current_time = std::chrono::high_resolution_clock::now() - start_time;
		float delta_time = current_time.count() - previous_frame_time;
        previous_frame_time = current_time.count();

        const uint8_t* keyboard_state = SDL_GetKeyboardState(nullptr);
        glm::vec3 kb_input = glm::zero<glm::vec3>();
        if(keyboard_state[SDL_SCANCODE_W]) kb_input += glm::vec3(0.0, 0.0, -1.0);
        if(keyboard_state[SDL_SCANCODE_A]) kb_input += glm::vec3(-1.0, 0.0, 0.0);
        if(keyboard_state[SDL_SCANCODE_S]) kb_input += glm::vec3(0.0, 0.0, 1.0);
        if(keyboard_state[SDL_SCANCODE_D]) kb_input += glm::vec3(1.0, 0.0, 0.0);
        if(keyboard_state[SDL_SCANCODE_Q]) kb_input += glm::vec3(0.0, 1.0, 0.0);
        if(keyboard_state[SDL_SCANCODE_E]) kb_input += glm::vec3(0.0, -1.0, 0.0);

        int mouse_dx, mouse_dy;
        SDL_GetRelativeMouseState(&mouse_dx, &mouse_dy);

        uint32_t mouse_state = SDL_GetMouseState(nullptr, nullptr);
        glm::vec3 world_up = glm::vec3(0.0, 1.0, 0.0);

        glm::vec3 camera_right = glm::normalize(glm::cross(camera_forward, world_up));
        if(SDL_BUTTON_RMASK & mouse_state) {
            camera_forward =
                camera_forward
                * glm::rotate(glm::identity<glm::quat>(), 0.5f * delta_time * (float)mouse_dx, world_up);
            camera_forward =
                camera_forward
                * glm::rotate(glm::identity<glm::quat>(), 0.5f * delta_time * (float)mouse_dy, camera_right);
        }
        camera_right = glm::normalize(glm::cross(camera_forward, world_up));
        glm::vec3 camera_up = glm::normalize(glm::cross(camera_right, camera_forward));

        glm::mat3 camera_basis = glm::mat3(camera_right, camera_up, -camera_forward);
        if(SDL_BUTTON_RMASK & mouse_state) {
            camera_position -= camera_basis * kb_input * delta_time * 10.0f;
        }

        glm::mat4 camera_rot = glm::mat4(glm::transpose(camera_basis));
        view_matrix = camera_rot * glm::translate(glm::identity<glm::mat4>(), camera_position);

        spaceship_object.orientation = glm::rotate(
            spaceship_object.orientation, 0.25f * delta_time, world_up
        );

        light_object.position = light_rot * glm::vec3(light_distance, 0.0, 0.0);
        glm::mat4 light_view_matrix = glm::lookAt(
            light_object.position, glm::zero<glm::vec3>(), world_up
        );
        glm::mat4 light_projection_matrix = glm::perspective(
            glm::radians(45.0f), 1.0f, 20.0f, 100.0f
        );
        light_projection_matrix[1][1] *= -1.0;

        global_ubo.light_matrix =
            glm::translate(glm::vec3(0.5f, 0.5f, 0.0f))
            * glm::scale(glm::vec3(0.5f, 0.5f, 1.0f))
            * light_projection_matrix * light_view_matrix
            * glm::inverse(view_matrix);

        global_ubo.view_inverse = glm::inverse(view_matrix);
        global_ubo.view_space_light_position = view_matrix * glm::vec4(light_object.position, 1.0f);
        global_ubo.light_view_dir = glm::normalize(
            glm::vec3(view_matrix * glm::vec4(-light_object.position, 0.0))
        );
        write_memory_mapped(vk_device, global_ubo_memory, global_ubo);

        bg_ubo.inv_pv = glm::inverse(projection_matrix * view_matrix);
        bg_ubo.camera_pos = camera_position;
        write_memory_mapped(vk_device, bg_ubo_memory, bg_ubo);

        ssao_ubo.inv_pv = glm::inverse(projection_matrix * view_matrix);
        ssao_ubo.projection_matrix = projection_matrix;
        ssao_ubo.inv_projection_matrix = glm::inverse(projection_matrix);
        write_memory_mapped(vk_device, ssao_ubo_memory, ssao_ubo);

        if(regenerate_shadowmap_pipeline) {
            vkDeviceWaitIdle(vk_device);
            /* vkDestroyPipeline(vk_device, shadowmap_pipeline, nullptr); */
            /* shadowmap_pipeline = create_graphics_pipeline( */
            /*     vk_device, */
            /*     shadowmap_pipeline_layout, */
            /*     render_pass, */
            /*     { model_attributes.first }, */
            /*     model_attributes.second, */
            /*     shadowmap_vert_shader_module, */
            /*     shadowmap_frag_shader_module, */
            /*     shadowmap_rasterizer */
            /* ); */
        }

        // ------------------------------------------------------------
        // Render frame
        // ------------------------------------------------------------

        vkWaitForFences(vk_device, 1, &frame_in_flight[current_frame], VK_TRUE, UINT64_MAX);

        uint32_t image_index;
        VkResult r = vkAcquireNextImageKHR(
            vk_device, swapchain, UINT64_MAX, image_avaiable[current_frame], VK_NULL_HANDLE, &image_index
        );
        if(r == VK_ERROR_OUT_OF_DATE_KHR) {
            surface_info = get_surface_info(physical_device, surface);
            vkDeviceWaitIdle(vk_device);
            recreate_swapchain(
                vk_device, physical_device,
                command_pool, graphics_queue,
                &swapchain, &swapchain_image_views,
                &depth_image, &depth_image_memory, &depth_image_view,
                &framebuffers,
                surface, surface_info,
                graphics_family, present_family,
                render_pass
            );
            continue;
        }

        vkResetFences(vk_device, 1, &frame_in_flight[current_frame]);

        // Render scene
        update_frame_data(
            vk_device,
            &frame_data,
            sampler, objects, models,
            view_matrix, projection_matrix
        );
        vkResetCommandBuffer(command_buffers[current_frame], 0);
        render_gbuffer(command_buffers[current_frame], image_index, current_frame);

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffers[current_frame];
        VK_HANDLE_ERROR(
            vkQueueSubmit(graphics_queue, 1, &submit_info, nullptr),
            "Could not submit queue"
        );
        vkDeviceWaitIdle(vk_device); // Too lazy to implement proper synchronization

        transition_image_layout(
            vk_device,
            command_pool, graphics_queue,
            gbuffer_depth_texture.m_image,
            depth_attachment.format,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1
        );
        
        transition_image_layout(
            vk_device,
            command_pool, graphics_queue,
            gbuffer_normals_texture.m_image,
            depth_attachment.format,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1
        );

        vkResetCommandBuffer(command_buffers[current_frame], 0);
        /* render_frame(command_buffers[current_frame], image_index, current_frame); */
        render_ssao(command_buffers[current_frame], image_index, current_frame);

        VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submit_info.pWaitDstStageMask = wait_stages;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &image_avaiable[current_frame];
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_finished[current_frame];

        VK_HANDLE_ERROR(
            vkQueueSubmit(graphics_queue, 1, &submit_info, frame_in_flight[current_frame]),
            "Could not submit queue"
        );
        vkDeviceWaitIdle(vk_device); // Too lazy to implement proper synchronization
        transition_image_layout(
            vk_device,
            command_pool, graphics_queue,
            gbuffer_depth_texture.m_image,
            depth_attachment.format,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1
        );
        transition_image_layout(
            vk_device,
            command_pool, graphics_queue,
            gbuffer_normals_texture.m_image,
            depth_attachment.format,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            1
        );

        // Present on screen
        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished[current_frame];
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain;
        present_info.pImageIndices = &image_index;

        r = vkQueuePresentKHR(present_queue, &present_info);
        vkDeviceWaitIdle(vk_device); // Too lazy to implement proper synchronization
        if(r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || framebuffer_resized) {
            framebuffer_resized = false;
            surface_info = get_surface_info(physical_device, surface);
            vkDeviceWaitIdle(vk_device);
            recreate_swapchain(
                vk_device, physical_device,
                command_pool, graphics_queue,
                &swapchain, &swapchain_image_views,
                &depth_image, &depth_image_memory, &depth_image_view,
                &framebuffers,
                surface, surface_info,
                graphics_family, present_family,
                render_pass
            );
            continue;
        }

        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    vkDeviceWaitIdle(vk_device);

    // ------------------------------------------------------------
    // Clean up
    // ------------------------------------------------------------

    /* vkDestroyShaderModule(vk_device, shadowmap_vert_shader_module, nullptr); */
    /* vkDestroyShaderModule(vk_device, shadowmap_frag_shader_module, nullptr); */
    for(auto& m : landingpad_model.m_materials) {
        if(m.m_name == "TV_Screen") m.m_emission_texture = {};
    }
    /* for(auto fb : shadowmap_framebuffers) vkDestroyFramebuffer(vk_device, fb, nullptr); */
    /* for(auto t : shadowmap_color_textures) t.destroy(vk_device); */
    /* for(auto t : shadowmap_depth_textures) t.destroy(vk_device); */
    vkDestroyDescriptorSetLayout(vk_device, global_descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, global_descriptor_pool, nullptr);
    vkDestroyBuffer(vk_device, global_ubo_buffer, nullptr);
    vkFreeMemory(vk_device, global_ubo_memory, nullptr);
    vkDestroyDescriptorSetLayout(vk_device, bg_descriptor_set_layout, nullptr);
    vkDestroyPipeline(vk_device, bg_pipeline, nullptr);
    vkDestroyPipelineLayout(vk_device, bg_pipeline_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, bg_descriptor_pool, nullptr);
    vkDestroyBuffer(vk_device, bg_ubo_buffer, nullptr);
    vkFreeMemory(vk_device, bg_ubo_memory, nullptr);
    vkDestroyBuffer(vk_device, quad_vertex_buffer, nullptr);
    vkFreeMemory(vk_device, quad_vertex_memory, nullptr);
    env_map.destroy(vk_device);
    irradiance_map.destroy(vk_device);
    reflection_map.destroy(vk_device);
    imgui_cleanup(vk_device, imgui_descriptor_pool);
    destroy_frame_data(vk_device, frame_data);
    for(auto& model : models) model->destroy(vk_device);
    vkDestroyDescriptorSetLayout(vk_device, material_descriptor_set_layout, nullptr);
    vkDestroySampler(vk_device, sampler, nullptr);
    /* vkDestroySampler(vk_device, shadow_sampler, nullptr); */
    for(auto f : frame_in_flight) vkDestroyFence(vk_device, f, nullptr);
    for(auto s : render_finished) vkDestroySemaphore(vk_device, s, nullptr);
    for(auto s : image_avaiable) vkDestroySemaphore(vk_device, s, nullptr);
    vkDestroyCommandPool(vk_device, command_pool, nullptr);
    for(const auto framebuffer : framebuffers) {
        vkDestroyFramebuffer(vk_device, framebuffer, nullptr);
    }
    /* vkDestroyPipeline(vk_device, shadowmap_pipeline, nullptr); */
    /* vkDestroyPipelineLayout(vk_device, shadowmap_pipeline_layout, nullptr); */
    vkDestroyPipeline(vk_device, material_pipeline, nullptr);
    vkDestroyPipelineLayout(vk_device, material_pipeline_layout, nullptr);
    vkDestroyRenderPass(vk_device, render_pass, nullptr);
    vkDestroyImage(vk_device, depth_image, nullptr);
    vkDestroyImageView(vk_device, depth_image_view, nullptr);
    vkFreeMemory(vk_device, depth_image_memory, nullptr);
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
