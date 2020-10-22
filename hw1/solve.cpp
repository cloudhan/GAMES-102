#include "solve.hpp"

#include <Eigen/Dense>
#include <iostream>

LeastSquare::LeastSquare()
    : m{-1} {}

LeastSquare::LeastSquare(int m, const std::vector<Point>& points)
    : m{m} {
    Matrixf A;
    Vectorf b;

    A.resize(points.size(), m + 1);
    b.resize(points.size());
    for (int i = 0; i < points.size(); i++) {
        A(i, 0) = 1.0;
        b(i)    = points[i].y;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = A(i, j - 1) * points[i].x;
        }
    }

    if (points.size()) {
        coeff = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    } else {
        coeff.resize(m + 1);
        coeff.setConstant(std::numeric_limits<float>::quiet_NaN());
    }
}

std::vector<Point> LeastSquare::predict(float x_start, float x_end, int num_points) {
    auto step = (x_end - x_start) / (num_points - 1);
    float x   = x_start;

    std::vector<Point> ret;
    ret.resize(num_points);

    Matrixf A(num_points, m + 1);
    for (int i = 0; i < num_points; i++, x += step) {
        ret[i].x = x;
        A(i, 0)  = 1.0;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = A(i, j - 1) * x;
        }
    }
    Vectorf ys = A * coeff;
    for (int i = 0; i < num_points; i++) {
        ret[i].y = ys(i);
    }
    return ret;
}

RidgeRegression::RidgeRegression()
    : m{-1} {}

RidgeRegression::RidgeRegression(int m, float a, const std::vector<Point>& points)
    : m{m}
    , a{a} {
    Matrixf A;
    Vectorf b;

    A.resize(points.size(), m + 1);
    b.resize(points.size());
    for (int i = 0; i < points.size(); i++) {
        A(i, 0) = 1.0;
        b(i)    = points[i].y;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = A(i, j - 1) * points[i].x;
        }
    }

    if (points.size()) {
        Matrixf AT_A = A.transpose() * A;
        std::cout << "before" << AT_A.diagonal() << std::endl;
        AT_A.diagonal().array() += a;
        std::cout << "After" << AT_A.diagonal() << std::endl;
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

std::vector<Point> RidgeRegression::predict(float x_start, float x_end, int num_points) {
    auto step = (x_end - x_start) / (num_points - 1);
    float x   = x_start;

    std::vector<Point> ret;
    ret.resize(num_points);

    Matrixf A(num_points, m + 1);
    for (int i = 0; i < num_points; i++, x += step) {
        ret[i].x = x;
        A(i, 0)  = 1.0;
        for (int j = 1; j < m + 1; j++) {
            A(i, j) = A(i, j - 1) * x;
        }
    }
    Vectorf ys = A * coeff;
    for (int i = 0; i < num_points; i++) {
        ret[i].y = ys(i);
    }
    return ret;
}
