#pragma once
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>

#define VK_HANDLE_ERROR(X, msg) if(X != VK_SUCCESS) { throw std::runtime_error(msg); }

const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

void init_vulkan(
    SDL_Window* window,
    VkInstance* vk_instance,
    VkDevice* vk_device, VkPhysicalDevice* physical_device,
    uint32_t* graphics_family, uint32_t* present_family,
    VkQueue* graphics_queue, VkQueue* present_queue,
    VkSurfaceKHR* surface
);
