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
        int num_points{150}; // control the smoothness of GUI
        MonomialInterpolation solver;
        bool enabled{false};  // is this solver enabled
        bool solve{true};     // should we solve the system
        bool predict{true};   // should we do the prediction pass
        vector<Point> points; // cached predicted points
    } monomial;

    struct
    {
        float sigma{33};
        int num_points{150};
        GaussInterpolation solver;
        bool enabled{false};
        bool solve{true};
        bool predict{true};
        vector<Point> points;
    } gauss;

    struct
    {
        int m{3};
        int num_points{150};
        LeastSquare solver;
        bool enabled{false};
        bool solve{true};
        bool predict{true};
        vector<Point> points;
    } least_square;

    struct
    {
        int m{3};
        float a{0.01};
        int num_points{150};
        RidgeRegression solver;
        bool enabled{false};
        bool solve{true};
        bool predict{true};
        vector<Point> points;
    } ridge_regression;
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
        ptr = gui_data.points_str.data() + (int)ptr;
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
        gui_data.monomial.solve = true;
        gui_data.gauss.solve = true;
        gui_data.least_square.solve = true;
        gui_data.ridge_regression.solve = true;
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
        ImGui::Checkbox("Enable Monomial Interpolation", &gui_data.monomial.enabled);
        ImGui::SameLine();
        ImGui::InputInt("Points##1", &gui_data.monomial.num_points, 1, 10);
        ImGui::EndGroup();

        ImGui::BeginGroup();
        ImGui::Checkbox("Enable Gauss Interpolation   ", &gui_data.gauss.enabled);
        ImGui::SameLine();
        ImGui::InputInt("Points##2", &gui_data.gauss.num_points, 1, 10);
        ImGui::SameLine();
        ImGui::InputFloat("Weight##4", &gui_data.gauss.sigma, 0.5, 5.0);
        ImGui::EndGroup();

        ImGui::BeginGroup();
        ImGui::Checkbox("Enable Least Square          ", &gui_data.least_square.enabled);
        ImGui::SameLine();
        ImGui::InputInt("Points##3", &gui_data.least_square.num_points, 1, 10);
        ImGui::SameLine();
        ImGui::InputInt("Order##3", &gui_data.least_square.m, 1, 1);
        ImGui::EndGroup();

        ImGui::BeginGroup();
        ImGui::Checkbox("Enable Ridge Regression      ", &gui_data.ridge_regression.enabled);
        ImGui::SameLine();
        ImGui::InputInt("Points##4", &gui_data.ridge_regression.num_points, 1, 10);
        ImGui::SameLine();
        ImGui::InputInt("Order##4", &gui_data.ridge_regression.m, 1, 1);
        ImGui::SameLine();
        ImGui::InputFloat("Weight##4", &gui_data.ridge_regression.a, 0.001, 0.01);
        ImGui::EndGroup();

        ImGui::Text("Mouse Right: drag to scroll, click for context menu.");

        if (gui_data.monomial.enabled) {
            auto& mi = gui_data.monomial;

            if (mi.num_points < 2)
                mi.num_points = 2;

            if (mi.num_points != mi.points.size())
                mi.predict = true;

            if (mi.solve) {
                mi.solver = MonomialInterpolation(gui_data.points);
                mi.solve = false;
                mi.predict = true;
            }
        }

        if (gui_data.gauss.enabled) {
            auto& gs = gui_data.gauss;
            if (gs.sigma < 10)
                gs.sigma = 10;
            if (gs.sigma > 100)
                gs.sigma = 100;

            if (gs.num_points < 2)
                gs.num_points = 2;

            if (gs.sigma != gs.solver.sigma)
                gs.solve = true;

            if (gs.num_points != gs.points.size())
                gs.predict = true;

            if (gs.solve) {
                gs.solver = GaussInterpolation(gs.sigma, gui_data.points);
                gs.solve = false;
                gs.predict = true;
            }
        }

        if (gui_data.least_square.enabled) {
            auto& ls = gui_data.least_square;
            if (ls.m < 0)
                ls.m = 0;
            if (ls.m > 15)
                ls.m = 15;
            if (ls.num_points < 2)
                ls.num_points = 2;

            if (ls.m != ls.solver.m)
                ls.solve = true;

            if (ls.num_points != ls.points.size())
                ls.predict = true;

            if (ls.solve) {
                ls.solver = LeastSquare(ls.m, gui_data.points);
                ls.solve = false;
                ls.predict = true;
            }
        }

        if (gui_data.ridge_regression.enabled) {
            auto& rr = gui_data.ridge_regression;
            if (rr.m < 0)
                rr.m = 0;
            if (rr.m > 15)
                rr.m = 15;

            if (rr.a < 0.001)
                rr.a = 0.001;
            if (rr.a > 1)
                rr.a = 1;

            if (rr.num_points < 2)
                rr.num_points = 2;

            if (rr.m != rr.solver.m) {
                rr.solve = true;
                rr.predict = true;
            }

            if (rr.a != rr.solver.a) {
                rr.solve = true;
                rr.predict = true;
            }

            if (rr.num_points != rr.points.size())
                rr.predict = true;

            if (rr.solve) {
                rr.solver = RidgeRegression(rr.m, rr.a, gui_data.points);
                rr.solve = false;
                rr.predict = true;
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

        if (gui_data.monomial.enabled && gui_data.monomial.solver.m > 0) {
            auto& mi = gui_data.monomial;
            if (mi.predict) {
                mi.points = mi.solver.predict(0, canvas_sz.x, mi.num_points);
                mi.predict = false;
            }
            const auto& xy = mi.points;
            for (int n = 1; n < xy.size(); n++) {
                draw_list->AddLine({origin.x + xy[n - 1].x, origin.y + xy[n - 1].y},
                                   {origin.x + xy[n].x, origin.y + xy[n].y},
                                   IM_COL32(128, 255, 255, 255), 2.0f);
            }
        }

        if (gui_data.gauss.enabled && gui_data.gauss.solver.m > 0) {
            auto& gs = gui_data.gauss;
            if (gs.predict) {
                gs.points = gs.solver.predict(0, canvas_sz.x, gs.num_points);
                gs.predict = false;
            }
            const auto& xy = gs.points;
            for (int n = 1; n < xy.size(); n++) {
                draw_list->AddLine({origin.x + xy[n - 1].x, origin.y + xy[n - 1].y},
                                   {origin.x + xy[n].x, origin.y + xy[n].y},
                                   IM_COL32(255, 128, 255, 255), 2.0f);
            }
        }

        if (gui_data.least_square.enabled) {
            auto& ll = gui_data.least_square;
            if (ll.predict) {
                ll.points = ll.solver.predict(0, canvas_sz.x, ll.num_points);
                ll.predict = false;
            }
            const auto& xy = ll.points;
            for (int n = 1; n < xy.size(); n++) {
                draw_list->AddLine({origin.x + xy[n - 1].x, origin.y + xy[n - 1].y},
                                   {origin.x + xy[n].x, origin.y + xy[n].y},
                                   IM_COL32(255, 255, 128, 255), 2.0f);
            }
        }

        if (gui_data.ridge_regression.enabled) {
            auto& rr = gui_data.ridge_regression;
            if (rr.predict) {
                rr.points = rr.solver.predict(0, canvas_sz.x, rr.num_points);
                rr.predict = false;
            }
            const auto& xy = rr.points;
            for (int n = 1; n < xy.size(); n++) {
                draw_list->AddLine({origin.x + xy[n - 1].x, origin.y + xy[n - 1].y},
                                   {origin.x + xy[n].x, origin.y + xy[n].y},
                                   IM_COL32(0, 255, 255, 255), 2.0f);
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
