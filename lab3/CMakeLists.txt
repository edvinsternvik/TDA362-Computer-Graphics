cmake_minimum_required(VERSION 3.15...3.30)

project(lab3)

add_executable(lab3 src/main.cpp)

target_compile_features(lab3 PUBLIC cxx_std_20)

target_include_directories(lab3 PRIVATE src)

target_link_libraries(lab3 PUBLIC labhelper)

add_custom_command(
    OUTPUT vert.spv
    COMMAND glslc src/shader.vert -o vert.spv
    DEPENDS src/shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab3"
)

add_custom_command(
    OUTPUT frag.spv
    COMMAND glslc src/shader.frag -o frag.spv
    DEPENDS src/shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lab3"
)
add_custom_target(generate_shaders_lab3
    DEPENDS vert.spv frag.spv
)

add_dependencies(lab3 generate_shaders_lab3)
