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

MonomialInterpolation::MonomialInterpolation(const std::vector<Point>& points)
    : norm{points}
    , m{static_cast<int>(points.size())}
{
    Matrixf A;
    Vectorf b;

    float x;
    float y;

    A.resize(m, m);
    b.resize(m);
    for (int i = 0; i < m; i++) {
        x = norm.normalize_x(points[i].x);
        y = norm.normalize_y(points[i].y);

        A(i, 0) = 1.0;
        b(i) = y;
        for (int j = 1; j < m; j++) {
            A(i, j) = A(i, j - 1) * x;
        }
    }

    if (m) {
        coeff = A.householderQr().solve(b);
    }
    else {
        coeff.resize(m);
        coeff.setConstant(std::numeric_limits<float>::quiet_NaN());
    }
}

std::vector<Point> MonomialInterpolation::predict(float x_start, float x_end, int num_points)
{
    auto step = (x_end - x_start) / (num_points - 1);
    float x = x_start;

    std::vector<Point> ret;
    ret.resize(num_points);

    Matrixf A(num_points, m);
    for (int i = 0; i < num_points; i++, x += step) {
        ret[i].x = x;

        float normalized_x = norm.normalize_x(x);
        A(i, 0) = 1.0;
        for (int j = 1; j < m; j++) {
            A(i, j) = A(i, j - 1) * normalized_x;
        }
    }
    Vectorf ys = A * coeff;
    for (int i = 0; i < num_points; i++) {
        ret[i].y = norm.denormalize_y(ys(i));
    }
    return ret;
}

namespace
{

/// g_i(x) is guass parameterized with x_i and evaluated at x.
/// each g_i is a basis.
float gauss(float x_i, float sigma, float x)
{
    return std::exp(-(x - x_i) * (x - x_i) / (2 * sigma * sigma));
}
} // namespace

GaussInterpolation::GaussInterpolation(float sigma, const std::vector<Point>& points)
    : m{static_cast<int>(points.size())}
    , sigma{sigma}
{
    xs.resize(points.size());
    for (int i = 0; i < m; i++) {
        // store them for prediction
        xs(i) = points[i].x;
    }

    if (points.size() == 1) {
        coeff.resize(m + 1);
        coeff(0) = points[0].y;
        coeff(1) = 0;
        return;
    }

    Matrixf A;
    Vectorf b;

    A.resize(m + 1, m + 1);
    b.resize(m + 1);
    for (int i = 0; i < m; i++) {
        float x_i = points[i].x;

        A(i, 0) = 1.0;
        b(i) = points[i].y;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = gauss(xs(j - 1), sigma, x_i);
        }
    }

    // Find the closest two point on x-axis and use the mid-point as last equation
    auto sorted_points = points;
    std::sort(sorted_points.begin(), sorted_points.end(),
              [](const Point& a, const Point& b) { return a.x < b.x; });

    int nearest = 0;
    float dist = sorted_points[nearest + 1].x - sorted_points[nearest].x;
    for (int i = 1; i < sorted_points.size(); i++) {
        auto new_dist = sorted_points[i].x - sorted_points[i - 1].x;
        if (new_dist < dist) {
            nearest = i - 1;
            dist = new_dist;
        }
    }

    float mid_x = (sorted_points[nearest + 1].x + sorted_points[nearest].x) * 0.5;
    float mid_y = (sorted_points[nearest + 1].y + sorted_points[nearest].y) * 0.5;

    A(m, 0) = 1.0;
    b(m) = mid_y;
    for (int j = 1; j < m + 1; j++) {
        A(m, j) = gauss(xs(j - 1), sigma, mid_x);
    }

    if (m) {
        coeff = A.householderQr().solve(b);
    }
    else {
        coeff.resize(m + 1);
        coeff.setConstant(std::numeric_limits<float>::quiet_NaN());
    }
}

std::vector<Point> GaussInterpolation::predict(float x_start, float x_end, int num_points)
{
    auto step = (x_end - x_start) / (num_points - 1);
    float x = x_start;

    std::vector<Point> ret;
    ret.resize(num_points);

    Matrixf A(num_points, m + 1);
    for (int i = 0; i < num_points; i++, x += step) {
        ret[i].x = x;

        A(i, 0) = 1.0;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = gauss(xs(j - 1), sigma, x);
        }
    }

    Vectorf ys = A * coeff;
    for (int i = 0; i < num_points; i++) {
        ret[i].y = ys(i);
    }
    return ret;
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
    }
    else {
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
        }
        else {
            coeff = AT_A.ldlt().solve(A.transpose() * b);
        }
    }
    else {
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
