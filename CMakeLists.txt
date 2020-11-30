cmake_minimum_required(VERSION 3.17)

project(GAMES_102 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- /wd4819")
endif()

set(IMGUI_SRCS
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_widgets.cpp
)

add_library(imgui ${IMGUI_SRCS})
target_include_directories(imgui PUBLIC "external/imgui")
target_include_directories(imgui PUBLIC "external/imgui/examples")

find_package(OpenGL REQUIRED)
find_package(glbinding REQUIRED)
find_package(glfw3 REQUIRED)

find_package(Eigen3 REQUIRED)

add_subdirectory(hw1)
add_subdirectory(hw2)
