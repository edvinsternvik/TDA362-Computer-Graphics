cmake_minimum_required(VERSION 3.15...3.30)

include(FetchContent)
FetchContent_Declare(
  stb
  GIT_REPOSITORY git@github.com:nothings/stb.git
  GIT_TAG        5c205738c191bcb0abc65c4febfa9bd25ff35234
)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY git@github.com:ocornut/imgui.git
  GIT_TAG        v1.91.5
)

FetchContent_MakeAvailable(stb imgui)

project(Lab1)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

find_package(SDL2 REQUIRED)
target_link_libraries(imgui PUBLIC SDL2 vulkan)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${SDL2_INCLUDE_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(lab1 src/main.cpp src/labhelper.cpp)

target_compile_features(lab1 PUBLIC cxx_std_20)

target_include_directories(lab1 PRIVATE
    src
    ${stb_SOURCE_DIR}
)

target_link_libraries(lab1 PUBLIC SDL2 vulkan imgui)

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
