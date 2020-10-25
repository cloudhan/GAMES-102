cmake_minimum_required(VERSION 3.17)

project(GAMES_102 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenGL REQUIRED)
find_package(glbinding REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)

find_package(Eigen3 REQUIRED)

add_subdirectory(hw1)
