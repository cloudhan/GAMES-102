set(SOURCES
    "hw1.cpp"
    "solve.cpp"
    "gui.cpp"
    "imgui_impl.cpp"
)

set(HEADERS
    "gui.hpp"
    "solve.hpp"
)

add_executable(hw1 ${SOURCES} ${HEADERS})
target_link_libraries(hw1 PRIVATE imgui::imgui glfw OpenGL::GL Eigen3::Eigen)
# target_link_libraries(hw1 PRIVATE glbinding::KHRplatform)
target_link_libraries(hw1 PRIVATE glbinding::glbinding)
# target_link_libraries(hw1 PRIVATE glbinding::glbinding-aux)
