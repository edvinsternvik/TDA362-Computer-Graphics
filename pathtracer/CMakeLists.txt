cmake_minimum_required(VERSION 3.15...3.30)

project(pathtracer)

find_package(embree 4 REQUIRED)
include_directories (${EMBREE_INCLUDE_DIRS})

find_package (OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(pathtracer
    src/main.cpp
    src/pathtracer.cpp
    src/sampling.cpp
    src/hdr_image.cpp
    src/embree.cpp
    src/material.cpp
)

target_compile_features(pathtracer PUBLIC cxx_std_20)

target_include_directories(pathtracer PRIVATE src)

target_link_libraries(pathtracer PUBLIC labhelper embree)

add_custom_command(
    OUTPUT vert.spv
    COMMAND glslc src/shader.vert -o vert.spv
    DEPENDS src/shader.vert
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/pathtracer"
)
add_custom_command(
    OUTPUT frag.spv
    COMMAND glslc src/shader.frag -o frag.spv
    DEPENDS src/shader.frag
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/pathtracer"
)
add_custom_target(generate_shaders_pathtracer
    DEPENDS vert.spv frag.spv
)

add_dependencies(pathtracer generate_shaders_pathtracer)
