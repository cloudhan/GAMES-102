#include "solve.hpp"

#include <Eigen/Dense>
#include <iostream>

Normalizer::Normalizer(const std::vector<Point>& points)
    : mean_x{0}
    , mean_y{0}
    , std_x{0}
    , std_y{0}
{
    for (const auto& p : points) {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= points.size();
    mean_y /= points.size();

    for (const auto& p : points) {
        std_x += (p.x - mean_x) * (p.x - mean_x);
        std_y += (p.y - mean_y) * (p.y - mean_y);
    }
    std_x /= points.size() - 1;
    std_y /= points.size() - 1;
    std_x = sqrt(std_x);
    std_y = sqrt(std_y);
}

float Normalizer::normalize_x(float x)
{
    return (x - mean_x) / std_x;
}

float Normalizer::normalize_y(float y)
{
    return (y - mean_y) / std_y;
}

float Normalizer::denormalize_x(float x)
{
    return x * std_x + mean_x;
}

float Normalizer::denormalize_y(float y)
{
    return y * std_y + mean_y;
}

LeastSquare::LeastSquare(int m, const std::vector<Point>& points)
    : norm{points}
    , m{m}
{
    Matrixf A;
    Vectorf b;

    float x;
    float y;

    A.resize(points.size(), m + 1);
    b.resize(points.size());
    for (int i = 0; i < points.size(); i++) {
        x = norm.normalize_x(points[i].x);
        y = norm.normalize_y(points[i].y);

        A(i, 0) = 1.0;
        b(i) = y;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = A(i, j - 1) * x;
        }
    }

    if (points.size()) {
        coeff = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    } else {
        coeff.resize(m + 1);
        coeff.setConstant(std::numeric_limits<float>::quiet_NaN());
    }
}

std::vector<Point> LeastSquare::predict(float x_start, float x_end, int num_points)
{
    auto step = (x_end - x_start) / (num_points - 1);
    float x = x_start;

    std::vector<Point> ret;
    ret.resize(num_points);

    Matrixf A(num_points, m + 1);
    for (int i = 0; i < num_points; i++, x += step) {
        ret[i].x = x;

        float normalized_x = norm.normalize_x(x);
        A(i, 0) = 1.0;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = A(i, j - 1) * normalized_x;
        }
    }
    Vectorf ys = A * coeff;
    for (int i = 0; i < num_points; i++) {
        ret[i].y = norm.denormalize_y(ys(i));
    }
    return ret;
}

RidgeRegression::RidgeRegression(int m, float a, const std::vector<Point>& points)
    : norm{points}
    , m{m}
    , a{a}
{
    Matrixf A;
    Vectorf b;

    float x;
    float y;

    A.resize(points.size(), m + 1);
    b.resize(points.size());
    for (int i = 0; i < points.size(); i++) {
        x = norm.normalize_x(points[i].x);
        y = norm.normalize_y(points[i].y);

        A(i, 0) = 1.0;
        b(i) = y;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = A(i, j - 1) * x;
        }
    }

    if (points.size()) {
        Matrixf AT_A = A.transpose() * A;
        AT_A.diagonal().array() += a;
        if (a > 0) {
            coeff = AT_A.llt().solve(A.transpose() * b);
        } else {
            coeff = AT_A.ldlt().solve(A.transpose() * b);
        }
    } else {
        coeff.resize(m + 1);
        coeff.setConstant(std::numeric_limits<float>::quiet_NaN());
    }
}

std::vector<Point> RidgeRegression::predict(float x_start, float x_end, int num_points)
{
    auto step = (x_end - x_start) / (num_points - 1);
    float x = x_start;

    std::vector<Point> ret;
    ret.resize(num_points);

    Matrixf A(num_points, m + 1);
    for (int i = 0; i < num_points; i++, x += step) {
        ret[i].x = x;

        float normalized_x = norm.normalize_x(x);
        A(i, 0) = 1.0;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = A(i, j - 1) * normalized_x;
        }
    }
    Vectorf ys = A * coeff;
    for (int i = 0; i < num_points; i++) {
        ret[i].y = norm.denormalize_y(ys(i));
    }
    return ret;
}
