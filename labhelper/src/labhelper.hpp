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
    uint32_t m_mip_levels;
};

Texture load_texture_from_image(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    const char* file_name
);

struct MaterialData {
    glm::vec3 m_color;
    float m_metalic;
    float m_fresnel;
    float m_roughness;
    float m_transparency;
    float m_ior;
    glm::vec3 m_emission;
};

struct Material {
    void destroy(VkDevice device);

    std::string m_name;
    MaterialData m_data;
    std::optional<Texture> m_color_texture;
    std::optional<Texture> m_metalic_texture;
    std::optional<Texture> m_fresnel_texture;
    std::optional<Texture> m_roughness_texture;
    std::optional<Texture> m_emission_texture;
};

struct Mesh {
    std::string m_name;
    size_t m_material_index;
    size_t m_start_index;
    size_t m_num_vertices;
};

struct Model {
    void destroy(VkDevice device);

    std::string m_name;
    std::vector<Material> m_materials;
    VkBuffer m_vertex_buffer;
    VkDeviceMemory m_vertex_buffer_memory;
    std::vector<Mesh> m_meshes;
};

Model load_model_from_file(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    const char* file_name
);

struct Object {
    size_t m_model_index;
    glm::vec3 position;
};

struct FrameData {
    VkDescriptorPool m_descriptor_pool;
    std::vector<VkBuffer> m_mvp_uniform_buffers;
    std::vector<VkDeviceMemory> m_mvp_uniform_memory;
    std::vector<VkBuffer> m_material_uniform_buffers;
    std::vector<VkDeviceMemory> m_material_uniform_memory;
    std::vector<VkDescriptorSet> m_descriptor_sets;
    VkImage m_empty_image;
    VkDeviceMemory m_empty_image_memory;
    VkImageView m_empty_image_view;
    size_t m_max_objects;
};

FrameData create_frame_data(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
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
    uint32_t* mip_levels,
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
    uint32_t width, uint32_t height,
    uint32_t mip_levels
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
    VkImageLayout old_layout, VkImageLayout new_layout,
    uint32_t mip_levels
);

void generate_mipmaps(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    VkImage image,
    uint32_t width, uint32_t height,
    uint32_t mip_levels,
    VkFormat format
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
