#include "gui.hpp"

#include <iostream>
#include <sstream>
#include <vector>

using namespace gl;
using namespace std;

void GlfwErrorCallback(int error, const char* description)
{
    cerr << "GLFW Error" << error << ": " << description << endl;
}

int main()
{
    glfwSetErrorCallback(GlfwErrorCallback);
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << endl;
        return 1;
    }

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "HW1", nullptr, nullptr);
    if (window == nullptr) {
        cerr << "Failed to create GLFW window" << endl;
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    glbinding::initialize(
        [](const char* name) { return (glbinding::ProcAddress)glfwGetProcAddress(name); });

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        DrawImGUI();
        glfwSwapBuffers(window);
    }
}
