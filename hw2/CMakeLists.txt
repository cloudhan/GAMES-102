set(SOURCES
    "hw2.cpp"
    "solve.cpp"
    "gui.cpp"
    "imgui_impl.cpp"
)

set(HEADERS
    "gui.hpp"
    "solve.hpp"
)

add_executable(hw2 ${SOURCES} ${HEADERS})
target_link_libraries(hw2 PRIVATE imgui glfw OpenGL::GL Eigen3::Eigen)
target_link_libraries(hw2 PRIVATE glbinding::glbinding)
