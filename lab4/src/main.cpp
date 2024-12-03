#include "SDL_keyboard.h"
#include "SDL_mouse.h"
#include "labhelper.hpp"
#include "model.hpp"
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/matrix.hpp>
#include <optional>
#include <vulkan/vulkan_core.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <imgui.h>
#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_vulkan.h>
#include <chrono>

const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

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

    // Material render pipeline

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


    // Specify descriptors
    struct GlobalUBO {
        glm::mat4 view_inverse;
        glm::vec3 view_space_light_position;
        float light_intensity;
        glm::vec3 light_color;
        float env_multiplier;
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
            0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
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
        }
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

    VkDescriptorSetLayout material_descriptor_set_layout =
        create_model_descriptor_set_layout(vk_device);

    // Specify vertex data description
    auto model_attributes = create_model_attributes();

    // Shader
    std::vector<char> material_vert_shader_src = read_file("lab4/vert.spv");
    std::vector<char> material_frag_shader_src = read_file("lab4/frag.spv");
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

    // Background render pipeline

    //
    VkVertexInputBindingDescription bg_binding_description = {};
    bg_binding_description.binding = 0;
    bg_binding_description.stride = 2 * sizeof(float);
    bg_binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription bg_position_attribute;
    bg_position_attribute.binding = 0;
    bg_position_attribute.location = 0;
    bg_position_attribute.offset = 0;
    bg_position_attribute.format = VK_FORMAT_R32G32_SFLOAT;

    // Shader
    std::vector<char> bg_vert_shader_src = read_file("lab4/bg_vert.spv");
    std::vector<char> bg_frag_shader_src = read_file("lab4/bg_frag.spv");
    VkShaderModule bg_vert_shader_module = create_shader_module(
        vk_device, bg_vert_shader_src
    );
    VkShaderModule bg_frag_shader_module = create_shader_module(
        vk_device, bg_frag_shader_src
    );

    // Load background
    std::array<float, 6 * 2> bg_vertices = {
        -1.0,  1.0,   1.0,  1.0,   1.0, -1.0,
         1.0, -1.0,  -1.0, -1.0,  -1.0,  1.0
    };

    VkBuffer bg_vertex_buffer = create_buffer(
        vk_device,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        bg_vertices.size() * sizeof(float)
    );
    VkDeviceMemory bg_vertex_memory = allocate_buffer_memory(
        vk_device, physical_device,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        bg_vertex_buffer
    );
    write_buffer_staged(
        vk_device, physical_device,
        graphics_queue, command_pool,
        bg_vertex_buffer,
        bg_vertices.data(), bg_vertices.size() * sizeof(float)
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

    VkPipelineDepthStencilStateCreateInfo bg_depth_stencil = {};
    bg_depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    bg_depth_stencil.depthTestEnable = VK_FALSE;
    bg_depth_stencil.depthWriteEnable = VK_FALSE;
    bg_depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    bg_depth_stencil.depthBoundsTestEnable = VK_FALSE;
    bg_depth_stencil.minDepthBounds = 0.0f;
    bg_depth_stencil.maxDepthBounds = 1.0f;
    bg_depth_stencil.stencilTestEnable = VK_FALSE;
    bg_depth_stencil.front = {};
    bg_depth_stencil.back = {};

    VkPipeline bg_pipeline = create_graphics_pipeline(
        vk_device,
        bg_pipeline_layout,
        render_pass,
        { bg_binding_description },
        { bg_position_attribute },
        bg_vert_shader_module,
        bg_frag_shader_module,
        bg_depth_stencil
    );

    vkDestroyShaderModule(vk_device, bg_vert_shader_module, nullptr);
    vkDestroyShaderModule(vk_device, bg_frag_shader_module, nullptr);

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
    std::vector<Model> models = {
        spaceship_model,
        materialtest_model,
        sphere_model
    };

    Object spaceship_object = {};
    spaceship_object.position = glm::vec3(0.0, 0.0, 0.0);
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

    std::vector<Object*> objects = {
        /* &spaceship_object, */
        &materialtest_object,
        &light_object
    };

    // Create uniform buffers
    glm::mat4 view_matrix = glm::identity<glm::mat4>();
    glm::mat4 projection_matrix = glm::perspective(glm::radians(45.0), (640.0 / 480.0), 0.01, 400.0);
    projection_matrix[1][1] *= -1.0;

    glm::vec3 view_space_light_pos = view_matrix * glm::vec4(light_object.position, 1.0);

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> frame_data;
    for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frame_data[i] = create_frame_data(
            vk_device, physical_device, command_pool,
            graphics_queue,
            material_descriptor_set_layout,
            100
        );

        update_frame_data(
            vk_device,
            &frame_data[i],
            sampler,
            objects,
            models,
            view_matrix,
            projection_matrix
        );
    }

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

    //

    uint32_t current_frame = 0;
    bool framebuffer_resized = false;
    glm::vec3 camera_position = -glm::vec3(-30.0, 10.0, 30.0);
    glm::vec3 camera_forward = glm::normalize(-glm::vec3(-30.0, 5.0, 30.0));

    auto start_time = std::chrono::high_resolution_clock::now();
    float previous_frame_time = 0.0f;

    BgUniformBlock bg_ubo = {};
    bg_ubo.inv_pv = glm::inverse(projection_matrix * view_matrix);
    bg_ubo.camera_pos = camera_position;
    bg_ubo.environment_multiplier = 1.0;
    write_memory_mapped(vk_device, bg_ubo_memory, bg_ubo);

    GlobalUBO global_ubo = {};
    global_ubo.view_inverse = glm::inverse(view_matrix);
    global_ubo.view_space_light_position = view_matrix * glm::vec4(light_object.position, 1.0);
    global_ubo.light_color = glm::vec3(1.0, 1.0, 1.0);
    global_ubo.light_intensity = 100.0;
    global_ubo.env_multiplier = 0.1;

    // Record command buffer for frame rendering
    glm::vec4 clear_color(0.0, 0.0, 0.0, 1.0);
    auto render_frame = [&](VkCommandBuffer command_buffer, uint32_t image_index, uint32_t current_frame) {
        update_frame_data(
            vk_device,
            &frame_data[current_frame],
            sampler, objects, models,
            view_matrix, projection_matrix
        );

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_HANDLE_ERROR(
            vkBeginCommandBuffer(command_buffer, &begin_info),
            "Could not begin command buffer"
        );

        std::array<VkClearValue, 2> clear_values = {};
        clear_values[0].color = { clear_color.r, clear_color.g, clear_color.b, clear_color.a };
        clear_values[1].depthStencil = { 1.0f, 0 };
        VkRenderPassBeginInfo render_pass_begin_info = {};
        render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_begin_info.renderPass = render_pass;
        render_pass_begin_info.framebuffer = framebuffers[image_index];
        render_pass_begin_info.renderArea.offset = {0, 0};
        render_pass_begin_info.renderArea.extent = surface_info.capabilities.currentExtent;
        render_pass_begin_info.clearValueCount = clear_values.size();
        render_pass_begin_info.pClearValues = clear_values.data();
        vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = surface_info.capabilities.currentExtent.width;
        viewport.height = surface_info.capabilities.currentExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = surface_info.capabilities.currentExtent;
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        VkDeviceSize offsets[] = { 0 };

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bg_pipeline);

        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            bg_pipeline_layout,
            0, 1,
            &bg_descriptor_set,
            0, nullptr
        );
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &bg_vertex_buffer, offsets);
        vkCmdDraw(command_buffer, bg_vertices.size(), 1, 0, 0);

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
            const Model& model = models[object->m_model_index];
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &model.m_vertex_buffer, offsets);

            for(const Mesh& mesh : model.m_meshes) {
                vkCmdBindDescriptorSets(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    material_pipeline_layout,
                    0, 1,
                    &frame_data[current_frame].m_descriptor_sets[descriptor_index],
                    0, nullptr
                );
                vkCmdDraw(command_buffer, mesh.m_num_vertices, 1, mesh.m_start_index, 0);

                descriptor_index++;
            }
        }

        imgui_new_frame();
        ImGui::SliderFloat("Light intensity", &global_ubo.light_intensity, 0.0, 500.0);
        ImGui::SliderFloat("Environment intensity", &global_ubo.env_multiplier, 0.0, 100.0);
        ImGui::SliderFloat("Environment multiplier", &bg_ubo.environment_multiplier, 0.0, 2.0);
        ImGui::ColorEdit3("Environment intensity", (float*)&global_ubo.light_color);
        imgui_render(command_buffer);

        vkCmdEndRenderPass(command_buffer);
        VK_HANDLE_ERROR(
            vkEndCommandBuffer(command_buffer),
            "Could not end command buffer"
        );

    };

    // Run application
    SDL_Event e; bool quit = false;
    while(quit == false) {
        while(SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if(e.type == SDL_QUIT) quit = true;
            if(e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED) framebuffer_resized = true;
        }

        // Game logic
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

        /* light_object.position = */ 
        /*     glm::rotate(glm::identity<glm::quat>(), delta_time * 10.0f, world_up) */
        /*     * light_object.position; */
        global_ubo.view_inverse = glm::inverse(view_matrix);
        global_ubo.view_space_light_position = view_matrix * glm::vec4(light_object.position, 1.0f);
        write_memory_mapped(vk_device, global_ubo_memory, global_ubo);

        bg_ubo.inv_pv = glm::inverse(projection_matrix * view_matrix);
        bg_ubo.camera_pos = camera_position;
        write_memory_mapped(vk_device, bg_ubo_memory, bg_ubo);

        // Render frame
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

        vkResetCommandBuffer(command_buffers[current_frame], 0);
        render_frame(command_buffers[current_frame], image_index, current_frame);

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

    // Clean up
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
    vkDestroyBuffer(vk_device, bg_vertex_buffer, nullptr);
    vkFreeMemory(vk_device, bg_vertex_memory, nullptr);
    env_map.destroy(vk_device);
    irradiance_map.destroy(vk_device);
    imgui_cleanup(vk_device, imgui_descriptor_pool);
    for(auto& fd : frame_data) destroy_frame_data(vk_device, fd);
    for(auto& model : models) model.destroy(vk_device);
    vkDestroyDescriptorSetLayout(vk_device, material_descriptor_set_layout, nullptr);
    vkDestroySampler(vk_device, sampler, nullptr);
    for(auto f : frame_in_flight) vkDestroyFence(vk_device, f, nullptr);
    for(auto s : render_finished) vkDestroySemaphore(vk_device, s, nullptr);
    for(auto s : image_avaiable) vkDestroySemaphore(vk_device, s, nullptr);
    vkDestroyCommandPool(vk_device, command_pool, nullptr);
    for(const auto framebuffer : framebuffers) {
        vkDestroyFramebuffer(vk_device, framebuffer, nullptr);
    }
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
