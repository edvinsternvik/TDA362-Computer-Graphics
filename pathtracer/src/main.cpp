#include "SDL_keyboard.h"
#include "SDL_mouse.h"
#include "embree.hpp"
#include "labhelper.hpp"
#include "model.hpp"
#include "pathtracer.hpp"
#include <algorithm>
#include <optional>
#include <vulkan/vulkan_core.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
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

    // Initialize path tracer
	pathtracer::settings.max_bounces = 8;
	pathtracer::settings.max_paths_per_pixel = 0; // 0 = Infinite
#ifdef _DEBUG
	pathtracer::settings.subsampling = 16;
#else
	pathtracer::settings.subsampling = 4;
#endif

	pathtracer::point_light.intensity_multiplier = 2500.0f;
	pathtracer::point_light.color = vec3(1.f, 1.f, 1.f);
	pathtracer::point_light.position = vec3(10.0f, 25.0f, 20.0f);

	pathtracer::environment.map.load("scenes/envmaps/001.hdr");
	pathtracer::environment.multiplier = 1.0f;

	pathtracer::restart();

    pathtracer::resize(
        surface_info.capabilities.currentExtent.width,
        surface_info.capabilities.currentExtent.height
    );


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

    //
    VkVertexInputBindingDescription binding_description = {};
    binding_description.binding = 0;
    binding_description.stride = 2 * sizeof(float);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription position_attribute;
    position_attribute.binding = 0;
    position_attribute.location = 0;
    position_attribute.offset = 0;
    position_attribute.format = VK_FORMAT_R32G32_SFLOAT;

    // Shader
    std::vector<char> vert_shader_src = read_file("pathtracer/vert.spv");
    std::vector<char> frag_shader_src = read_file("pathtracer/frag.spv");
    VkShaderModule vert_shader_module = create_shader_module(
        vk_device, vert_shader_src
    );
    VkShaderModule frag_shader_module = create_shader_module(
        vk_device, frag_shader_src
    );

    // Load background
    std::array<float, 6 * 2> vertices = {
        -1.0,  1.0,   1.0,  1.0,   1.0, -1.0,
         1.0, -1.0,  -1.0, -1.0,  -1.0,  1.0
    };

    VkBuffer vertex_buffer = create_buffer(
        vk_device,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        vertices.size() * sizeof(float)
    );
    VkDeviceMemory vertex_memory = allocate_buffer_memory(
        vk_device, physical_device,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vertex_buffer
    );
    write_buffer_staged(
        vk_device, physical_device,
        graphics_queue, command_pool,
        vertex_buffer,
        vertices.data(), vertices.size() * sizeof(float)
    );
    
    // Descriptors
    VkImage pathtrace_image = create_image(
        vk_device,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL,
        pathtracer::rendered_image.width,
        pathtracer::rendered_image.height, 1
    );
    VkDeviceMemory pathtrace_image_memory = allocate_image_memory(
        vk_device, physical_device, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        pathtrace_image
    );
    VkImageView pathtrace_image_view = create_image_view(
        vk_device,
        VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, 1,
        pathtrace_image
    );
    transition_image_layout(
        vk_device,
        command_pool, graphics_queue,
        pathtrace_image,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        1
    );
    std::vector<DescriptorInfo> descriptors = {
        {
            0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
            std::nullopt, std::nullopt,
            std::make_optional(pathtrace_image_view), std::make_optional(sampler)
        }
    };

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;
    create_descriptors(
        vk_device,
        &descriptor_pool,
        &descriptor_set_layout, &descriptor_set,
        descriptors
    );

    update_descriptors(vk_device, descriptor_set, descriptors);

    // Pipeline
    VkPipelineLayout pipeline_layout = create_pipeline_layout(
        vk_device,
        { descriptor_set_layout }
    );

    VkPipeline pipeline = create_graphics_pipeline(
        vk_device,
        pipeline_layout,
        render_pass,
        { binding_description },
        { position_attribute },
        vert_shader_module,
        frag_shader_module,
        std::nullopt,
        std::nullopt
    );

    vkDestroyShaderModule(vk_device, vert_shader_module, nullptr);
    vkDestroyShaderModule(vk_device, frag_shader_module, nullptr);

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
    std::vector<Model*> models = {
        &spaceship_model,
        &materialtest_model,
        &sphere_model
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
        &spaceship_object,
        /* &materialtest_object, */
        &light_object
    };

    // Add objects to pathtracer
	pathtracer::reinit_scene();

	for(const auto& o : objects) {
        glm::mat4 model_matrix = glm::identity<glm::mat4>()
            * glm::translate(glm::identity<glm::mat4>(), o->position)
            * glm::mat4_cast(o->orientation)
            * glm::scale(glm::identity<glm::mat4>(), o->scale);

		pathtracer::add_model(models[o->m_model_index], model_matrix);
	}
	pathtracer::build_bvh();

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

    // Record command buffer for frame rendering
    glm::vec4 clear_color(0.0, 0.0, 0.0, 1.0);
    auto render_frame = [&](VkCommandBuffer command_buffer, uint32_t image_index, uint32_t current_frame) {
        mat4 viewMatrix = lookAt(-camera_position, -camera_position + camera_forward, glm::vec3(0, 1, 0));
        mat4 projMatrix = perspective(
            radians(45.0f),
            float(pathtracer::rendered_image.width) / float(pathtracer::rendered_image.height), 
            0.1f, 100.0f
        );
        projMatrix[1][1] *= -1.0;
        pathtracer::trace_paths(viewMatrix, projMatrix);

        size_t image_size = pathtracer::rendered_image.width * pathtracer::rendered_image.height * 4;
        std::vector<uint8_t> image_data(image_size, 0);
        for(size_t pxl = 0; pxl < image_size / 4; ++pxl) {
            image_data[pxl * 4 + 0] = 255 * std::clamp(pathtracer::rendered_image.data[pxl].r, 0.0f, 1.0f);
            image_data[pxl * 4 + 1] = 255 * std::clamp(pathtracer::rendered_image.data[pxl].g, 0.0f, 1.0f);
            image_data[pxl * 4 + 2] = 255 * std::clamp(pathtracer::rendered_image.data[pxl].b, 0.0f, 1.0f);
            image_data[pxl * 4 + 3] = 255;
        }
        transition_image_layout(
            vk_device,
            command_pool, graphics_queue,
            pathtrace_image,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1
        );
        write_image_staged(
            vk_device, physical_device,
            graphics_queue, command_pool,
            pathtrace_image,
            image_data.data(), image_size,
            pathtracer::rendered_image.width, pathtracer::rendered_image.height, 0
        );
        transition_image_layout(
            vk_device,
            command_pool, graphics_queue,
            pathtrace_image,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1
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

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0, 1,
            &descriptor_set,
            0, nullptr
        );
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, offsets);
        vkCmdDraw(command_buffer, vertices.size(), 1, 0, 0);

        imgui_new_frame();
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
    vkDestroyImage(vk_device, pathtrace_image, nullptr);
    vkDestroyImageView(vk_device, pathtrace_image_view, nullptr);
    vkFreeMemory(vk_device, pathtrace_image_memory, nullptr);
    vkDestroyDescriptorSetLayout(vk_device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, descriptor_pool, nullptr);
    vkDestroyPipeline(vk_device, pipeline, nullptr);
    vkDestroyPipelineLayout(vk_device, pipeline_layout, nullptr);
    vkDestroyBuffer(vk_device, vertex_buffer, nullptr);
    vkFreeMemory(vk_device, vertex_memory, nullptr);
    imgui_cleanup(vk_device, imgui_descriptor_pool);
    for(auto& model : models) model->destroy(vk_device);
    vkDestroySampler(vk_device, sampler, nullptr);
    for(auto f : frame_in_flight) vkDestroyFence(vk_device, f, nullptr);
    for(auto s : render_finished) vkDestroySemaphore(vk_device, s, nullptr);
    for(auto s : image_avaiable) vkDestroySemaphore(vk_device, s, nullptr);
    vkDestroyCommandPool(vk_device, command_pool, nullptr);
    for(const auto framebuffer : framebuffers) {
        vkDestroyFramebuffer(vk_device, framebuffer, nullptr);
    }
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
