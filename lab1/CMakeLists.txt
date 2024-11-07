cmake_minimum_required(VERSION 3.15...3.30)

project(Lab1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(lab1 src/main.cpp)

target_compile_features(lab1 PUBLIC cxx_std_20)

target_include_directories(lab1 PRIVATE src)

target_link_libraries(lab1 PUBLIC SDL2 vulkan)

add_custom_command(
    OUTPUT vert.spv
    COMMAND glslc src/shader.vert -o vert.spv
    DEPENDS src/shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_custom_command(
    OUTPUT frag.spv
    COMMAND glslc src/shader.frag -o frag.spv
    DEPENDS src/shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
add_custom_target(generate_shaders
    DEPENDS vert.spv frag.spv
)

add_dependencies(lab1 generate_shaders)
