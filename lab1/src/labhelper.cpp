#include "labhelper.hpp"
#include <set>

#ifdef NDEBUG
const std::array<const char*, 0> VALIDATION_LAYERS = { };
#else
const std::array<const char*, 1> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};
#endif

const std::array<const char*, 1> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

void init_vulkan(
    SDL_Window* window,
    VkInstance* vk_instance,
    VkDevice* vk_device, VkPhysicalDevice* physical_device,
    uint32_t* graphics_family, uint32_t* present_family,
    VkQueue* graphics_queue, VkQueue* present_queue,
    VkSurfaceKHR* surface
) {
    // Check for validation layers
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
    if(!VALIDATION_LAYERS.empty() && !validation_layers_found) {
        throw std::runtime_error("Could not find validation layers");
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

    VK_HANDLE_ERROR(
        vkCreateInstance(&inst_create_info, nullptr, vk_instance),
        "Could not create instance"
    );

    // Create surface
    SDL_Vulkan_CreateSurface(window, *vk_instance, surface);

    // Get physical devices
    uint32_t phys_device_count;
    vkEnumeratePhysicalDevices(*vk_instance, &phys_device_count, nullptr);
    std::vector<VkPhysicalDevice> physical_devices(phys_device_count);
    vkEnumeratePhysicalDevices(*vk_instance, &phys_device_count, physical_devices.data());

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
            vkGetPhysicalDeviceSurfaceSupportKHR(curr_phys_device, i, *surface, &present_support);
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

        // Check for supported features
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(curr_phys_device, &features);
        if(!features.samplerAnisotropy) continue;

        *graphics_family = current_graphics_family.value();
        *present_family = current_present_family.value();
        *physical_device = curr_phys_device;
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
    device_features.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.enabledExtensionCount = DEVICE_EXTENSIONS.size();
    device_create_info.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    device_create_info.queueCreateInfoCount = queue_create_infos.size();
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.pEnabledFeatures = &device_features;

    VK_HANDLE_ERROR(
        vkCreateDevice(*physical_device, &device_create_info, nullptr, vk_device),
        "Could not create device"
    );

    // Get queues
    vkGetDeviceQueue(*vk_device, *graphics_family, 0, graphics_queue);
    vkGetDeviceQueue(*vk_device, *present_family, 0, present_queue);
}
