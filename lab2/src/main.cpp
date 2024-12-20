#include "labhelper.hpp"
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/trigonometric.hpp>
#include <vulkan/vulkan_core.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>
#include <imgui.h>
#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_vulkan.h>

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

    // Create shader
    std::vector<char> vert_shader_src = read_file("lab2/vert.spv");
    std::vector<char> frag_shader_src = read_file("lab2/frag.spv");
    VkShaderModule vert_shader_module = create_shader_module(vk_device, vert_shader_src);
    VkShaderModule frag_shader_module = create_shader_module(vk_device, frag_shader_src);

    // Specify descriptors
    VkDescriptorSetLayoutBinding ubo_layout_binding = {};
    ubo_layout_binding.binding = 0;
    ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo_layout_binding.descriptorCount = 1;
    ubo_layout_binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding sampler_layout_binding = {};
    sampler_layout_binding.binding = 1;
    sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sampler_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sampler_layout_binding.descriptorCount = 1;
    sampler_layout_binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayout descriptor_set_layout = create_descriptor_set_layout(
        vk_device,
        { ubo_layout_binding, sampler_layout_binding }
    );

    // Specify vertex data description
    VkVertexInputBindingDescription binding_description = {};
    binding_description.binding = 0;
    binding_description.stride = 5 * sizeof(float);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription position_attribute;
    position_attribute.binding = 0;
    position_attribute.location = 0;
    position_attribute.offset = 0;
    position_attribute.format = VK_FORMAT_R32G32B32_SFLOAT;

    VkVertexInputAttributeDescription uv_attribute;
    uv_attribute.binding = 0;
    uv_attribute.location = 1;
    uv_attribute.offset = 3 * sizeof(float);
    uv_attribute.format = VK_FORMAT_R32G32_SFLOAT;

    // Pipeline
    VkPipelineLayout pipeline_layout = create_pipeline_layout(
        vk_device,
        { descriptor_set_layout }
    );

    VkPipeline graphics_pipeline = create_graphics_pipeline(
        vk_device,
        pipeline_layout,
        render_pass,
        { binding_description },
        { position_attribute, uv_attribute },
        vert_shader_module,
        frag_shader_module
    );

    vkDestroyShaderModule(vk_device, vert_shader_module, nullptr);
    vkDestroyShaderModule(vk_device, frag_shader_module, nullptr);

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

    // Create vertex buffer
    std::vector<float> vertices[2];
    vertices[0] = {
    //     X     Y        Z     U     V
         10.0f, 0.0f,  -10.0f, 1.0f, 0.0f,
        -10.0f, 0.0f, -330.0f, 0.0f, 15.0f,
        -10.0f, 0.0f,  -10.0f, 0.0f, 0.0f,

         10.0f, 0.0f,  -10.0f, 1.0f, 0.0f,
         10.0f, 0.0f, -330.0f, 1.0f, 15.0f,
        -10.0f, 0.0f, -330.0f, 0.0f, 15.0f
    };

    vertices[1] = {
         0.5,  0.5, 0.0, 1.0, 1.0,
         0.5, -0.5, 0.0, 1.0, 0.0,
        -0.5, -0.5, 0.0, 0.0, 0.0,

        -0.5, -0.5, 0.0, 0.0, 0.0,
        -0.5,  0.5, 0.0, 0.0, 1.0,
         0.5,  0.5, 0.0, 1.0, 1.0
    };

    std::array<VkBuffer, 2> vertex_buffers;
    std::array<VkDeviceMemory, 2> vertex_buffer_memories;

    for(int i = 0; i < 2; ++i) {
        vertex_buffers[i] = create_buffer(
            vk_device,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            sizeof(float) * vertices[i].size()
        );

        vertex_buffer_memories[i] = allocate_buffer_memory(
            vk_device, physical_device,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertex_buffers[i]
        );

        write_buffer_staged(
            vk_device, physical_device,
            graphics_queue,
            command_pool,
            vertex_buffers[i],
            vertices[i].data(), sizeof(float) * vertices[i].size()
        );
    }

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

    // Create uniform buffers
    struct Object {
        Texture texture;
        VkBuffer vertex_buffer;
        VkDeviceMemory vertex_buffer_memory;
        glm::vec3 position;
    };
    Object asphalt = {};
    asphalt.texture = load_texture_from_image(
        vk_device, physical_device, command_pool, graphics_queue,
        "scenes/textures/asphalt.jpg"
    );
    asphalt.vertex_buffer = vertex_buffers[0];
    asphalt.vertex_buffer_memory = vertex_buffer_memories[0];
    asphalt.position = glm::vec3(0.0, 10.0, 0.0);
    Object explosion = {};
    explosion.texture = load_texture_from_image(
        vk_device, physical_device, command_pool, graphics_queue,
        "scenes/textures/explosion.png"
    );
    explosion.vertex_buffer = vertex_buffers[1];
    explosion.vertex_buffer_memory = vertex_buffer_memories[1];
    explosion.position = glm::vec3(0.5, 0.5, -4.0);
    std::array<Object*, 2> objects = {
        &asphalt, &explosion
    };

    glm::mat4 view_projection_matrix = glm::perspective(glm::radians(45.0), (640.0 / 480.0), 0.01, 400.0);

    struct FrameData {
        VkDescriptorPool descriptor_pool;
        std::vector<VkDescriptorSet> descriptor_sets;
        std::vector<VkBuffer> uniform_buffers;
        std::vector<VkDeviceMemory> uniform_memory;
    };

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> frame_data;
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frame_data[i].uniform_buffers.resize(objects.size());
        frame_data[i].uniform_memory.resize(objects.size());
        frame_data[i].descriptor_sets.resize(objects.size());

        std::array<VkDescriptorPoolSize, 2> descriptor_pool_sizes = {};
        descriptor_pool_sizes[0].descriptorCount = objects.size();
        descriptor_pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_pool_sizes[1].descriptorCount = objects.size();
        descriptor_pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

        VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
        descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool_create_info.maxSets = objects.size();
        descriptor_pool_create_info.poolSizeCount = descriptor_pool_sizes.size();
        descriptor_pool_create_info.pPoolSizes = descriptor_pool_sizes.data();
        vkCreateDescriptorPool(vk_device, &descriptor_pool_create_info, nullptr, &frame_data[i].descriptor_pool);

        std::vector<VkDescriptorSetLayout> descriptor_set_layouts(objects.size(), descriptor_set_layout);
        VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
        descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptor_set_allocate_info.descriptorPool = frame_data[i].descriptor_pool;
        descriptor_set_allocate_info.descriptorSetCount = objects.size();
        descriptor_set_allocate_info.pSetLayouts = descriptor_set_layouts.data();
        vkAllocateDescriptorSets(vk_device, &descriptor_set_allocate_info, frame_data[i].descriptor_sets.data());

        for(size_t j = 0; j < objects.size(); ++j) {
            frame_data[i].uniform_buffers[j] = create_buffer(
                vk_device,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                sizeof(glm::mat4)
            );
            frame_data[i].uniform_memory[j] = allocate_buffer_memory(
                vk_device, physical_device,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                frame_data[i].uniform_buffers[j]
            );
        }
    }

    auto update_frame_data = [&](uint32_t frame_index) {
        for(size_t obj_i = 0; obj_i < objects.size(); ++obj_i) {
            VkDescriptorBufferInfo descriptor_buffer_info = {};
            descriptor_buffer_info.buffer = frame_data[frame_index].uniform_buffers[obj_i];
            descriptor_buffer_info.range = sizeof(glm::mat4);
            descriptor_buffer_info.offset = 0;

            VkDescriptorImageInfo descriptor_image_info = {};
            descriptor_image_info.sampler = sampler;
            descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            descriptor_image_info.imageView = objects[obj_i]->texture.m_image_view;

            std::array<VkWriteDescriptorSet, 2> descriptor_set_writes = {};
            descriptor_set_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_set_writes[0].descriptorCount = 1;
            descriptor_set_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_set_writes[0].dstSet = frame_data[frame_index].descriptor_sets[obj_i];
            descriptor_set_writes[0].dstBinding = 0;
            descriptor_set_writes[0].dstArrayElement = 0;
            descriptor_set_writes[0].pBufferInfo = &descriptor_buffer_info;
            descriptor_set_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_set_writes[1].descriptorCount = 1;
            descriptor_set_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptor_set_writes[1].dstSet = frame_data[frame_index].descriptor_sets[obj_i];
            descriptor_set_writes[1].dstBinding = 1;
            descriptor_set_writes[1].dstArrayElement = 0;
            descriptor_set_writes[1].pImageInfo = &descriptor_image_info;
            vkUpdateDescriptorSets(
                vk_device,
                descriptor_set_writes.size(),
                descriptor_set_writes.data(),
                0, nullptr
            );

            glm::mat4* mvp_mapping;
            vkMapMemory(
                vk_device,
                frame_data[frame_index].uniform_memory[obj_i],
                0,
                sizeof(glm::mat4),
                0,
                (void**)&mvp_mapping
            );

            *((glm::mat4*)mvp_mapping) =
                view_projection_matrix
                * glm::translate(glm::identity<glm::mat4>(), objects[obj_i]->position);

            vkUnmapMemory(vk_device, frame_data[frame_index].uniform_memory[obj_i]);
        }
    };

    auto destroy_frame_data = [&](const FrameData& frame_data) {
        vkDestroyDescriptorPool(vk_device, frame_data.descriptor_pool, nullptr);
        for(auto ub : frame_data.uniform_buffers) {
            vkDestroyBuffer(vk_device, ub, nullptr);
        };
        for(auto um : frame_data.uniform_memory) {
            vkFreeMemory(vk_device, um, nullptr);
        }
    };
    auto destroy_object = [&](Object& object) {
        object.texture.destroy(vk_device);
        vkDestroyBuffer(vk_device, object.vertex_buffer, nullptr);
        vkFreeMemory(vk_device, object.vertex_buffer_memory, nullptr);
    };

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

    // Record command buffer for frame rendering
    glm::vec4 clear_color(0.0, 0.0, 0.0, 1.0);
    VkFilter mag_filter = sampler_create_info.magFilter;
    VkFilter min_filter = sampler_create_info.minFilter;
    VkSamplerMipmapMode mipmap_mode = sampler_create_info.mipmapMode;
    float anisotropy = sampler_create_info.maxAnisotropy;
    auto render_frame = [&](VkCommandBuffer command_buffer, uint32_t image_index, uint32_t current_frame) {
        update_frame_data(current_frame);

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

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        VkDeviceSize offsets[] = { 0 };

        for(size_t obj_i = 0; obj_i < objects.size(); ++obj_i) {
            Object* object = objects[obj_i];
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &object->vertex_buffer, offsets);
            
            vkCmdBindDescriptorSets(
                command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout,
                0, 1,
                &frame_data[current_frame].descriptor_sets[obj_i],
                0, nullptr
            );

            vkCmdDraw(command_buffer, vertices[obj_i].size(), 1, 0, 0);
        }

        imgui_new_frame();
        ImGui::ColorEdit4("clear colour", (float*)&clear_color);

        ImGui::PushID("mag");
        ImGui::Text("Mag filter");
        ImGui::RadioButton("NEAREST", (int*)&mag_filter, VK_FILTER_NEAREST);
        ImGui::RadioButton("LINEAR", (int*)&mag_filter, VK_FILTER_LINEAR);
        ImGui::PopID();

        ImGui::PushID("min");
        ImGui::Text("Min filter");
        ImGui::RadioButton("NEAREST", (int*)&min_filter, VK_FILTER_NEAREST);
        ImGui::RadioButton("LINEAR", (int*)&min_filter, VK_FILTER_LINEAR);
        ImGui::PopID();

        ImGui::PushID("mipmap");
        ImGui::Text("Mipmap mode");
        ImGui::RadioButton("NEAREST", (int*)&mipmap_mode, VK_SAMPLER_MIPMAP_MODE_NEAREST);
        ImGui::RadioButton("LINEAR", (int*)&mipmap_mode, VK_SAMPLER_MIPMAP_MODE_LINEAR);
        ImGui::PopID();

        ImGui::SliderFloat("Anisotropic filtering", &anisotropy, 1.0, device_properties.limits.maxSamplerAnisotropy, "Number of samples: %.0f");
        imgui_render(command_buffer);

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

    // Run application
    SDL_Event e; bool quit = false;
    while(quit == false) {
        while(SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if(e.type == SDL_QUIT) quit = true;
            if(e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED) framebuffer_resized = true;
        }

        bool sampler_changed =
            mag_filter != sampler_create_info.magFilter
            || min_filter != sampler_create_info.minFilter
            || mipmap_mode != sampler_create_info.mipmapMode
            || anisotropy != sampler_create_info.maxAnisotropy;
        if(sampler_changed) {
            vkDeviceWaitIdle(vk_device);
            vkDestroySampler(vk_device, sampler, nullptr);

            sampler_create_info.magFilter = mag_filter;
            sampler_create_info.minFilter = min_filter;
            sampler_create_info.mipmapMode = mipmap_mode;
            sampler_create_info.maxAnisotropy = anisotropy;
            vkCreateSampler(vk_device, &sampler_create_info, nullptr, &sampler);
        }

        // Render frame
        vkWaitForFences(vk_device, 1, &frame_in_flight[current_frame], VK_TRUE, UINT64_MAX);

        uint32_t image_index;
        VkResult r = vkAcquireNextImageKHR(vk_device, swapchain, UINT64_MAX, image_avaiable[current_frame], VK_NULL_HANDLE, &image_index);
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
    imgui_cleanup(vk_device, imgui_descriptor_pool);
    for(auto& obj : objects) destroy_object(*obj);
    for(auto& fd : frame_data) destroy_frame_data(fd);
    vkDestroyDescriptorSetLayout(vk_device, descriptor_set_layout, nullptr);
    vkDestroySampler(vk_device, sampler, nullptr);
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
