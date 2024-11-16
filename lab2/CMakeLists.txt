cmake_minimum_required(VERSION 3.15...3.30)

project(lab2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(lab2 src/main.cpp)

target_compile_features(lab2 PUBLIC cxx_std_20)

target_include_directories(lab2 PRIVATE src)

target_link_libraries(lab2 PUBLIC labhelper)

add_custom_command(
    OUTPUT vert.spv
    COMMAND glslc src/shader.vert -o vert.spv
    DEPENDS src/shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab2"
)

add_custom_command(
    OUTPUT frag.spv
    COMMAND glslc src/shader.frag -o frag.spv
    DEPENDS src/shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab2"
)
add_custom_target(generate_shaders_lab2
    DEPENDS vert.spv frag.spv
)

add_dependencies(lab2 generate_shaders_lab2)
