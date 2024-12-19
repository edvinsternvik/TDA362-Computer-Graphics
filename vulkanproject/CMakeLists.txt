cmake_minimum_required(VERSION 3.15...3.30)

project(vulkanproject)

add_executable(vulkanproject src/main.cpp)

target_compile_features(vulkanproject PUBLIC cxx_std_20)

target_include_directories(vulkanproject PRIVATE src)

target_link_libraries(vulkanproject PUBLIC labhelper)

add_custom_command(
    OUTPUT vert.spv
    COMMAND glslc src/shader.vert -o vert.spv
    DEPENDS src/shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT frag.spv
    COMMAND glslc src/shader.frag -o frag.spv
    DEPENDS src/shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT bg_vert.spv
    COMMAND glslc src/bg_shader.vert -o bg_vert.spv
    DEPENDS src/bg_shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT bg_frag.spv
    COMMAND glslc src/bg_shader.frag -o bg_frag.spv
    DEPENDS src/bg_shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT depth_normal_vert.spv
    COMMAND glslc src/depth_normal_shader.vert -o depth_normal_vert.spv
    DEPENDS src/depth_normal_shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT depth_normal_frag.spv
    COMMAND glslc src/depth_normal_shader.frag -o depth_normal_frag.spv
    DEPENDS src/depth_normal_shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT shadowmap_vert.spv
    COMMAND glslc src/shadowmap_shader.vert -o shadowmap_vert.spv
    DEPENDS src/shadowmap_shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT shadowmap_frag.spv
    COMMAND glslc src/shadowmap_shader.frag -o shadowmap_frag.spv
    DEPENDS src/shadowmap_shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_target(generate_shaders_vulkanproject
    DEPENDS vert.spv frag.spv bg_vert.spv bg_frag.spv depth_normal_vert.spv depth_normal_frag.spv shadowmap_vert.spv shadowmap_frag.spv
)

add_dependencies(vulkanproject generate_shaders_vulkanproject)
