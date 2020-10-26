#pragma once

#include "gui.hpp"

#include "Eigen/Core"

using Matrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vectorf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

struct Normalizer
{
    float mean_x;
    float mean_y;
    float std_x;
    float std_y;
    Normalizer() {}
    Normalizer(const std::vector<Point>& points);

    float normalize_x(float x);
    float normalize_y(float y);

    float denormalize_x(float x);
    float denormalize_y(float y);
};

struct MonomialInterpolation
{
    Normalizer norm;
    int m;         // the highest order of the basis function
    Vectorf coeff; // the solved coefficient
    MonomialInterpolation() {}
    MonomialInterpolation(const std::vector<Point>& points);
    std::vector<Point> predict(float x_start, float x_end, int num_points);
};

struct GaussInterpolation
{
    int m;         // number of Gauss basis
    float sigma;   // the global standard deviation
    Vectorf coeff; // the solved coefficient
    Vectorf xs;    // the original xs that form the basis functions
    GaussInterpolation() {}
    GaussInterpolation(float sigma, const std::vector<Point>& points);
    std::vector<Point> predict(float x_start, float x_end, int num_points);
};

struct LeastSquare
{
    Normalizer norm;
    int m;         // the highest order of the basis function
    Vectorf coeff; // the solved coefficient
    LeastSquare() {}
    LeastSquare(int m, const std::vector<Point>& points);
    std::vector<Point> predict(float x_start, float x_end, int num_points);
};

struct RidgeRegression
{
    Normalizer norm;
    int m;
    float a; // the weighting term of normalization
    Vectorf coeff;
    RidgeRegression() {}
    RidgeRegression(int m, float a, const std::vector<Point>& points);
    std::vector<Point> predict(float x_start, float x_end, int num_points);
};
