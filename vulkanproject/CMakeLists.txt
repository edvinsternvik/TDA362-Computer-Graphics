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
    OUTPUT gbuffer_vert.spv
    COMMAND glslc src/gbuffer_shader.vert -o gbuffer_vert.spv
    DEPENDS src/gbuffer_shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT gbuffer_frag.spv
    COMMAND glslc src/gbuffer_shader.frag -o gbuffer_frag.spv
    DEPENDS src/gbuffer_shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT ssao_vert.spv
    COMMAND glslc src/ssao_shader.vert -o ssao_vert.spv
    DEPENDS src/ssao_shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/vulkanproject"
)
add_custom_command(
    OUTPUT ssao_frag.spv
    COMMAND glslc src/ssao_shader.frag -o ssao_frag.spv
    DEPENDS src/ssao_shader.frag
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
    DEPENDS vert.spv frag.spv bg_vert.spv bg_frag.spv gbuffer_vert.spv gbuffer_frag.spv ssao_vert.spv ssao_frag.spv shadowmap_vert.spv shadowmap_frag.spv
)

add_dependencies(vulkanproject generate_shaders_vulkanproject)
