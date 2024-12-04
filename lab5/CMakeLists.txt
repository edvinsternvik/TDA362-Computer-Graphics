cmake_minimum_required(VERSION 3.15...3.30)

project(lab5)

add_executable(lab5 src/main.cpp)

target_compile_features(lab5 PUBLIC cxx_std_20)

target_include_directories(lab5 PRIVATE src)

target_link_libraries(lab5 PUBLIC labhelper)

add_custom_command(
    OUTPUT vert.spv
    COMMAND glslc src/shader.vert -o vert.spv
    DEPENDS src/shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab5"
)
add_custom_command(
    OUTPUT frag.spv
    COMMAND glslc src/shader.frag -o frag.spv
    DEPENDS src/shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab5"
)
add_custom_command(
    OUTPUT bg_vert.spv
    COMMAND glslc src/bg_shader.vert -o bg_vert.spv
    DEPENDS src/bg_shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab5"
)
add_custom_command(
    OUTPUT bg_frag.spv
    COMMAND glslc src/bg_shader.frag -o bg_frag.spv
    DEPENDS src/bg_shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab5"
)
add_custom_command(
    OUTPUT postfx_vert.spv
    COMMAND glslc src/postfx_shader.vert -o postfx_vert.spv
    DEPENDS src/postfx_shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab5"
)
add_custom_command(
    OUTPUT postfx_frag.spv
    COMMAND glslc src/postfx_shader.frag -o postfx_frag.spv
    DEPENDS src/postfx_shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab5"
)
add_custom_target(generate_shaders_lab5
    DEPENDS vert.spv frag.spv bg_vert.spv bg_frag.spv postfx_vert.spv postfx_frag.spv
)

add_dependencies(lab5 generate_shaders_lab5)
