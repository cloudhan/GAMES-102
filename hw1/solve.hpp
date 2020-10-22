#pragma once

#include "gui.hpp"

#include "Eigen/Core"

using Matrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vectorf = Eigen::Matrix<float, Eigen::Dynamic, 1>;


// basis function: 1, x, x^2, ...

struct LeastSquare {
    int m; // the highest order of the basis function
    Vectorf coeff; // the solved coefficient
    LeastSquare();
    LeastSquare(int m, const std::vector<Point>& points);
    std::vector<Point> predict(float x_start, float x_end, int num_points);
};

struct RidgeRegression {
    int m;
    float a; // the weighting term of normalization
    Vectorf coeff;
    RidgeRegression();
    RidgeRegression(int m, float a, const std::vector<Point>& points);
    std::vector<Point> predict(float x_start, float x_end, int num_points);
};
