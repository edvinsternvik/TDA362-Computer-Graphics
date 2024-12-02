#pragma once
#include "labhelper.hpp"
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vulkan/vulkan_core.h>

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
    glm::quat orientation;
    glm::vec3 scale;
};

struct FrameData {
    VkDescriptorPool m_descriptor_pool;
    std::vector<VkBuffer> m_mvp_uniform_buffers;
    std::vector<VkDeviceMemory> m_mvp_uniform_memory;
    std::vector<VkBuffer> m_material_uniform_buffers;
    std::vector<VkDeviceMemory> m_material_uniform_memory;
    std::vector<VkDescriptorSet> m_descriptor_sets;
    std::vector<std::vector<VkBuffer>> m_uniform_buffers;
    std::vector<std::vector<VkDeviceMemory>> m_uniform_memory;
    VkImage m_empty_image;
    VkDeviceMemory m_empty_image_memory;
    VkImageView m_empty_image_view;
    size_t m_max_objects;
};

struct Descriptor {
    VkShaderStageFlags stage_flags;
    uint32_t size;
};

FrameData create_frame_data(
    VkDevice device, VkPhysicalDevice physical_device,
    VkCommandPool command_pool, VkQueue command_queue,
    VkDescriptorSetLayout descriptor_set_layout,
    const size_t max_objects,
    const std::vector<Descriptor>& descriptors
);

void update_frame_data(
    VkDevice device,
    FrameData* frame_data,
    VkSampler sampler,
    const std::vector<Object*>& objects,
    const std::vector<Model>& models,
    glm::mat4 view_matrix,
    glm::mat4 projection_matrix,
    const std::vector<Descriptor>& descriptors,
    const std::vector<void*>& descr_write_data
);

void destroy_frame_data(
    VkDevice device,
    FrameData& frame_data
);

VkDescriptorSetLayout create_model_descriptor_set_layout(
    VkDevice device,
    const std::vector<Descriptor>& descriptors
);

std::pair<
    VkVertexInputBindingDescription,
    std::vector<VkVertexInputAttributeDescription>
> create_model_attributes();
