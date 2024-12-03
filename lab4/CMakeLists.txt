cmake_minimum_required(VERSION 3.15...3.30)

project(lab4)

add_executable(lab4 src/main.cpp)

target_compile_features(lab4 PUBLIC cxx_std_20)

target_include_directories(lab4 PRIVATE src)

target_link_libraries(lab4 PUBLIC labhelper)

add_custom_command(
    OUTPUT vert.spv
    COMMAND glslc src/shader.vert -o vert.spv
    DEPENDS src/shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab4"
)
add_custom_command(
    OUTPUT frag.spv
    COMMAND glslc src/shader.frag -o frag.spv
    DEPENDS src/shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab4"
)
add_custom_command(
    OUTPUT bg_vert.spv
    COMMAND glslc src/bg_shader.vert -o bg_vert.spv
    DEPENDS src/bg_shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab4"
)
add_custom_command(
    OUTPUT bg_frag.spv
    COMMAND glslc src/bg_shader.frag -o bg_frag.spv
    DEPENDS src/bg_shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab4"
)
add_custom_target(generate_shaders_lab4
    DEPENDS vert.spv frag.spv bg_vert.spv bg_frag.spv
)

add_dependencies(lab4 generate_shaders_lab4)
