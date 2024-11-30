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
FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY git@github.com:tinyobjloader/tinyobjloader.git
  GIT_TAG        5cd3842fdca3b06cc993801cff1825fc6d999068
)

FetchContent_MakeAvailable(stb imgui tinyobjloader)

project(labhelper)

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
target_include_directories(imgui PUBLIC
    ${CMAKE_SOURCE_DIR}/labhelper
    ${imgui_SOURCE_DIR}
    ${SDL2_INCLUDE_DIR}
)

add_library(labhelper src/labhelper.cpp src/model.cpp)

target_compile_features(labhelper PUBLIC cxx_std_20)

target_include_directories(labhelper PUBLIC
    src
    ${stb_SOURCE_DIR}
)

target_link_libraries(labhelper PUBLIC SDL2 vulkan imgui tinyobjloader)
