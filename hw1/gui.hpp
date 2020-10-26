#pragma once

#define IMGUI_IMPL_OPENGL_LOADER_GLBINDING3
#define GLFW_INCLUDE_NONE 1
#include <imgui.h>

#if __has_include("bindings/imgui_impl_glfw.h")
#define USE_VCPKG 1
#endif

#if USE_VCPKG
// headers from vcpkg
#include "bindings/imgui_impl_glfw.h"
#include "bindings/imgui_impl_opengl3.h"
#else
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#endif

#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>

#include <GLFW/glfw3.h>

struct Point
{
    float x;
    float y;
};

void DrawImGUI();
