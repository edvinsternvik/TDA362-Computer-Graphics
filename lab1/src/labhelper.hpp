#pragma once
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>

#define VK_HANDLE_ERROR(X, msg) if(X != VK_SUCCESS) { throw std::runtime_error(msg); }

const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

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
