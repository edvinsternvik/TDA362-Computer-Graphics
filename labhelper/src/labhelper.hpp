#pragma once
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan_core.h>
#include <stb_image.h>

#include <glm/glm.hpp>

#define VK_HANDLE_ERROR(X, msg) if(X != VK_SUCCESS) { throw std::runtime_error(msg); }

const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct Texture {
    void destroy(VkDevice device);

    VkImage m_image;
    VkImageView m_image_view;
    VkDeviceMemory m_image_memory;
    int m_width, m_height, m_channels;
};

Texture load_texture_from_image(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    const char* file_name
);

struct Material {
    Texture m_texture;
};

struct Mesh {
    size_t m_material_index;
    size_t m_vertex_offset;
    size_t m_num_vertices;
};

struct Model {
    void destroy(VkDevice device);

    Material m_material;
    VkBuffer m_vertex_buffer;
    VkDeviceMemory m_vertex_buffer_memory;
    VkBuffer m_index_buffer;
    VkDeviceMemory m_index_buffer_memory;
};

struct Object {
    size_t m_model_index;
    glm::vec3 position;
};

struct FrameData {
    VkDescriptorPool m_descriptor_pool;
    std::vector<VkBuffer> m_uniform_buffers;
    std::vector<VkDeviceMemory> m_uniform_memory;
    std::vector<VkDescriptorSet> m_descriptor_sets;
    size_t m_max_objects;
};

FrameData create_frame_data(
    VkDevice device, VkPhysicalDevice physical_device,
    VkDescriptorSetLayout descriptor_set_layout,
    const size_t max_objects
);

void update_frame_data(
    VkDevice device,
    FrameData* frame_data,
    VkSampler sampler,
    const std::vector<Object>& objects,
    const std::vector<Model>& models,
    glm::mat4 view_projection_matrix
);

void destroy_frame_data(
    VkDevice device,
    FrameData& frame_data
);

struct SurfaceInfo {
    VkSurfaceFormatKHR format;
    VkPresentModeKHR present_mode;
    VkSurfaceCapabilitiesKHR capabilities;
};

void init_vulkan(
    SDL_Window* window,
    VkInstance* vk_instance,
    VkDevice* vk_device, VkPhysicalDevice* physical_device,
    uint32_t* graphics_family, uint32_t* present_family,
    VkQueue* graphics_queue, VkQueue* present_queue,
    VkCommandPool* command_pool,
    VkSurfaceKHR* surface
);

VkRenderPass create_render_pass(
    VkDevice device,
    VkAttachmentDescription color_attachment,
    VkAttachmentDescription depth_attachment
);

VkDescriptorSetLayout create_descriptor_set_layout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayoutBinding>& layout_bindings
);

VkPipelineLayout create_pipeline_layout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts
);

VkPipeline create_graphics_pipeline(
    VkDevice device,
    VkPipelineLayout pipeline_layout,
    VkRenderPass render_pass,
    const std::vector<VkVertexInputBindingDescription>& vertex_binding_descriptions,
    const std::vector<VkVertexInputAttributeDescription>& vertex_attribute_descriptions,
    VkShaderModule vert_shader_module,
    VkShaderModule frag_shader_module
);

void load_image(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    VkImage* image, VkDeviceMemory* memory,
    const char* file_name,
    int* width, int* height, int* channels,
    int required_comp,
    VkImageUsageFlags usage,
    VkFormat format,
    VkImageTiling tiling,
    VkMemoryPropertyFlags memory_properties
);

SurfaceInfo get_surface_info(
    VkPhysicalDevice physical_device,
    VkSurfaceKHR surface
);

std::vector<char> read_file(
    const char* file_name
);

VkShaderModule create_shader_module(
    VkDevice device,
    std::vector<char>& shader_src
);

VkCommandBuffer begin_single_use_command(
    VkDevice device,
    VkCommandPool command_pool
);

void submit_command(
    VkQueue cmd_queue,
    VkCommandBuffer cmd_buffer
);

VkBuffer create_buffer(
    VkDevice device,
    VkBufferUsageFlags usage,
    size_t buffer_size
);

VkImage create_image(
    VkDevice device,
    VkImageUsageFlags usage,
    VkFormat format,
    VkImageTiling tiling,
    uint32_t width, uint32_t height
);

uint32_t get_suitable_memory_type_index(
    VkPhysicalDevice physical_device,
    VkMemoryRequirements mem_requirements,
    VkMemoryPropertyFlags properties
);

VkDeviceMemory allocate_buffer_memory(
    VkDevice device, VkPhysicalDevice physical_device,
    VkMemoryPropertyFlags properties,
    VkBuffer buffer
);

void write_buffer_staged(
    VkDevice device, VkPhysicalDevice physical_device, VkQueue cmd_queue,
    VkCommandPool command_pool,
    VkBuffer buffer,
    void* data, size_t data_size
);

VkSwapchainKHR create_swapchain(
    VkDevice device,
    VkSurfaceKHR surface,
    SurfaceInfo surface_info,
    uint32_t graphics_family,
    uint32_t present_family
);

std::vector<VkImageView> create_swapchain_image_views(
    VkDevice device,
    VkSwapchainKHR swapchain,
    VkFormat image_format
);

VkDeviceMemory allocate_image_memory(
    VkDevice device, VkPhysicalDevice physical_device,
    VkMemoryPropertyFlags properties,
    VkImage image
);

void transition_image_layout(
    VkDevice device,
    VkCommandPool command_pool,
    VkQueue command_queue,
    VkImage image,
    VkFormat format,
    VkImageLayout old_layout, VkImageLayout new_layout
);

void create_depth_buffer(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    uint32_t width, uint32_t height,
    VkImage* depth_image,
    VkDeviceMemory* depth_image_memory,
    VkImageView* depth_image_view
);

VkFramebuffer create_framebuffer(
    VkDevice device,
    VkRenderPass render_pass,
    VkImageView image_view,
    VkImageView depth_view,
    VkExtent2D extent
);

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
);

void write_image_staged(
    VkDevice device, VkPhysicalDevice physical_device, VkQueue cmd_queue,
    VkCommandPool command_pool,
    VkImage image,
    void* data, size_t data_size, uint32_t width, uint32_t height
);

VkDescriptorPool imgui_init(
    VkInstance instance,
    SDL_Window* window,
    VkDevice device, VkPhysicalDevice physical_device,
    uint32_t queue_family, VkQueue queue,
    VkRenderPass render_pass
);

void imgui_new_frame();
void imgui_render(VkCommandBuffer command_buffer);
void imgui_cleanup(VkDevice device, VkDescriptorPool imgui_descriptor_pool);
