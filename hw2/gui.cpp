#include "gui.hpp"
#include "solve.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

struct Vec2
{
    float x;
    float y;
};

struct GuiData
{
    vector<Point> points;
    string points_str{};
    int selected{0};
    vector<char*> points_str_view{};
    bool points_changed{false};
    Vec2 scrolling{0.f, 0.f};
    bool opt_enable_grid{true};
    bool opt_enable_context_menu{true};
    bool deleting_guard{false};

    struct
    {
        int num_basis{4};
        int num_points{150};
        std::shared_ptr<Optimizer> opt{new SgdOptimizer(0.1)};
        RBFNetwork solver{num_basis};
        bool enabled{true};
        bool fit{false};
        bool predict{false};
        vector<Point> points;
    } rbf;
};

GuiData gui_data{};

void GuiOnPointsChanged()
{
    if (!gui_data.points_changed)
        return;

    gui_data.points_str.clear();
    gui_data.points_str_view.clear();

    if (gui_data.points.size() == 0)
        return;

    ostringstream oss{};
    auto beg = oss.tellp();

    gui_data.points_str_view.reserve(gui_data.points.size());
    for (const auto& p : gui_data.points) {
        gui_data.points_str_view.push_back(reinterpret_cast<char*>(oss.tellp() - beg));
        oss << "x: " << p.x << ", "
            << "y:" << p.y;
        oss.put(0);
    }
    gui_data.points_str = oss.str();

    for (auto& ptr : gui_data.points_str_view) {
        ptr = gui_data.points_str.data() + reinterpret_cast<ptrdiff_t>(ptr);
    }
}

void DrawImGUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::IsKeyReleased(ImGui::GetKeyIndex(ImGuiKey_Delete))) {
        gui_data.deleting_guard = false;
    }

    if (gui_data.selected >= gui_data.points.size()) {
        gui_data.selected = gui_data.points.size() - 1;
    }

    if (!gui_data.deleting_guard && gui_data.points.size() > 0 &&
        ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Delete))) {
        gui_data.points.erase(gui_data.points.begin() + gui_data.selected);
        gui_data.points_changed = true;
        gui_data.deleting_guard = true;
    }

    if (gui_data.points_changed) {
        GuiOnPointsChanged();
        if (gui_data.points.size() > 0) {
            gui_data.rbf.fit = true;
        }
        gui_data.points_changed = false;
    }

    if (ImGui::Begin("Points")) {
        ImGui::ListBox("##1", &gui_data.selected, gui_data.points_str_view.data(),
                       gui_data.points_str_view.size(),
                       std::min<int>(30, gui_data.points_str_view.size()));
    }
    ImGui::End();

    if (ImGui::Begin("Canvas")) {
        ImGui::Checkbox("Enable grid", &gui_data.opt_enable_grid);
        ImGui::Checkbox("Enable context menu", &gui_data.opt_enable_context_menu);

        ImGui::PushItemWidth(100);

        ImGui::BeginGroup();
        if(ImGui::Button("RBF Network")) {
            gui_data.rbf.fit = true;
        }
        ImGui::SameLine();
        ImGui::InputInt("Points##1", &gui_data.rbf.num_points, 1, 10);
        ImGui::SameLine();
        ImGui::InputInt("Number of Basis##1", &gui_data.rbf.num_basis, 1, 10);
        ImGui::EndGroup();

        ImGui::Text("Mouse Right: drag to scroll, click for context menu.");

        if (gui_data.rbf.enabled) {
            auto& rbf = gui_data.rbf;

            if (rbf.num_points < 2)
                rbf.num_points = 2;

            if (rbf.num_points != rbf.points.size())
                rbf.predict = true;

            if (rbf.fit) {
                rbf.solver.fit(rbf.opt, gui_data.points);
                rbf.fit = false;
                rbf.predict = true;
            }
        }

        // Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2)
        // allows us to use IsItemHovered()/IsItemActive()
        ImVec2 canvas_sz = ImGui::GetContentRegionAvail(); // Resize canvas to what's available
        if (canvas_sz.x < 50.0f)
            canvas_sz.x = 50.0f;
        if (canvas_sz.y < 50.0f)
            canvas_sz.y = 50.0f;
        ImVec2 canvas_p0 = ImGui::GetCursorScreenPos(); // ImDrawList API uses screen coordinates!
        ImVec2 canvas_p1{canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y};

        // Draw border and background color
        ImGuiIO& io = ImGui::GetIO();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
        draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

        // This will catch our interactions
        ImGui::InvisibleButton("canvas", canvas_sz);
        const bool is_hovered = ImGui::IsItemHovered(); // Hovered
        const bool is_active = ImGui::IsItemActive();   // Held
        const ImVec2 origin(canvas_p0.x + gui_data.scrolling.x,
                            canvas_p0.y + gui_data.scrolling.y); // Lock scrolled origin
        const Point mouse_pos_in_canvas{io.MousePos.x - origin.x, io.MousePos.y - origin.y};

        // Add first and second point
        if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            gui_data.points.push_back(mouse_pos_in_canvas);
            gui_data.points_changed = true;
        }

        // Pan (we use a zero mouse threshold when there's no context menu)
        // You may decide to make that threshold dynamic based on whether the mouse is hovering
        // something etc.
        const float mouse_threshold_for_pan = gui_data.opt_enable_context_menu ? -1.0f : 0.0f;
        if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan)) {
            gui_data.scrolling.x += io.MouseDelta.x;
            gui_data.scrolling.y += io.MouseDelta.y;
        }

        // Context menu (under default mouse threshold)
        ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
        if (gui_data.opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) &&
            drag_delta.x == 0.0f && drag_delta.y == 0.0f)
            ImGui::OpenPopupOnItemClick("context");
        if (ImGui::BeginPopup("context")) {
            if (ImGui::MenuItem("Remove all", NULL, false, gui_data.points.size() > 0)) {
                gui_data.points.clear();
                gui_data.points_changed = true;
            }
            ImGui::EndPopup();
        }

        // Draw grid + all lines in the canvas
        draw_list->PushClipRect(canvas_p0, canvas_p1, true);
        if (gui_data.opt_enable_grid) {
            const float GRID_STEP = 64.0f;
            for (float x = fmodf(gui_data.scrolling.x, GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
                draw_list->AddLine({canvas_p0.x + x, canvas_p0.y}, {canvas_p0.x + x, canvas_p1.y},
                                   IM_COL32(200, 200, 200, 40));
            for (float y = fmodf(gui_data.scrolling.y, GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
                draw_list->AddLine({canvas_p0.x, canvas_p0.y + y}, {canvas_p1.x, canvas_p0.y + y},
                                   IM_COL32(200, 200, 200, 40));
        }

        if (gui_data.points.size()) {
            draw_list->AddCircleFilled(
                {origin.x + gui_data.points[0].x, origin.y + gui_data.points[0].y}, 4,
                IM_COL32(255, 100, 100, 255));
        }

        if (gui_data.rbf.enabled) {
            auto& rbf = gui_data.rbf;
            if (rbf.predict) {
                rbf.points = rbf.solver.predict(0, canvas_sz.x, rbf.num_points);
                rbf.predict = false;
            }
            const auto& xy = rbf.points;
            for (int n = 1; n < xy.size(); n++) {
                draw_list->AddLine({origin.x + xy[n - 1].x, origin.y + xy[n - 1].y},
                                   {origin.x + xy[n].x, origin.y + xy[n].y},
                                   IM_COL32(128, 255, 255, 255), 2.0f);
            }
        }


        for (int n = 1; n < gui_data.points.size(); n++) {
            draw_list->AddCircleFilled(
                {origin.x + gui_data.points[n].x, origin.y + gui_data.points[n].y}, 5,
                IM_COL32(255, 100, 100, 255));
        }
        draw_list->PopClipRect();
    }

    ImGui::End();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
