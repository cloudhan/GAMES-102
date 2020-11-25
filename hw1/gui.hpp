#pragma once

#define IMGUI_IMPL_OPENGL_LOADER_GLBINDING3
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#define GLFW_INCLUDE_NONE 1
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>

#include <GLFW/glfw3.h>

struct Point
{
    float x;
    float y;
};

void DrawImGUI();
