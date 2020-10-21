#pragma once

#include "gui.hpp"

#include "Eigen/Core"

using Matrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vectorf = Eigen::Matrix<float, Eigen::Dynamic, 1>;


struct LeastSquare {
    int m;
    Vectorf coeff;
    LeastSquare();
    LeastSquare(int m, const std::vector<Point>& points);
    std::vector<Point> predict(float x_start, float x_end, int num_points);
};
