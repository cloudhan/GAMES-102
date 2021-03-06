#include "solve.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <optional>
#include <random>

using std::vector;

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

SgdOptimizer::SgdOptimizer(float lr)
    : Optimizer(lr)
{
}

void SgdOptimizer::init_state(const vector<Matrixf*>& params) {}

void SgdOptimizer::update_params(const std::vector<Matrixf*>& params, const vector<Matrixf>& grads)
{
    for (int i = 0; i < params.size(); i++) {
        *params[i] -= lr * grads[i];
    }
}

AdamOptimizer::AdamOptimizer(float lr, float b1, float b2, float eps)
    : Optimizer{lr}
    , b1{b1}
    , b2{b2}
    , b1_{}
    , b2_{}
    , eps{eps}
    , m{}
    , v{}
    , step{}
{
}

void AdamOptimizer::init_state(const vector<Matrixf*>& params)
{
    step = 0;
    b1_ = 1;
    b2_ = 1;
    m.clear();
    v.clear();
    for(auto p : params) {
        m.emplace_back(Matrixf::Zero(p->rows(), p->cols()));
        v.emplace_back(Matrixf::Zero(p->rows(), p->cols()));
    }
}

void AdamOptimizer::update_params(const std::vector<Matrixf*>& params, const vector<Matrixf>& grads)
{
    step += 1;
    b1_ *= b1;
    b2_ *= b2;
    for(int i=0; i<grads.size(); i++) {
        m[i] = b1 * m[i] + (1.0f - b1) * grads[i];
        v[i] = b2 * v[i] + (1.0f - b2) * (grads[i].cwiseProduct(grads[i]));
        Matrixf d = (m[i] / (1.0 - b1_)).array() / ((v[i] / (1.0 - b2_)).cwiseSqrt().array() + eps);
        *params[i] -= lr * d;
    }
}

RBFNetwork::RBFNetwork(int num_basis)
    : num_basis{num_basis}
{
    w1.resize(1, num_basis);
    b1.resize(1, num_basis);
    w2.resize(num_basis, 1);
    b2.resize(1, 1);
}

void RBFNetwork::init(std::shared_ptr<Optimizer> opt)
{
    std::mt19937 engine{};
    std::normal_distribution<float> rand{};
    for (int c = 0; c < w1.cols(); c++) {
        for (int r = 0; r < w1.rows(); r++) {
            w1(r, c) = 0.2 * rand(engine);
        }
    }
    std::cout << w1 << std::endl;
    b1.setZero();

    for (int c = 0; c < w2.cols(); c++) {
        for (int r = 0; r < w2.rows(); r++) {
            w2(r, c) = 0.2 * rand(engine);
        }
    }
    std::cout << w2.transpose() << std::endl;
    b2.setZero();

    opt->init_state(vector{&w1, &b1, &w2, &b2});
}

// I do not have a tape, so it is not a AutoDiffFunc
using ManualDiffFunc = std::function<vector<std::optional<Matrixf>>(const Matrixf&)>;

std::tuple<Matrixf, ManualDiffFunc> fc(const Eigen::Ref<const Matrixf>& x,
                                       const Eigen::Ref<const Matrixf>& w,
                                       const Eigen::Ref<const Matrixf>& b)
{
    Matrixf ret = (x * w).rowwise() + b.row(0);

    auto grad_func = [=](const Matrixf& g) {
        Matrixf gx = g * w.transpose();
        Matrixf gw = x.transpose() * g;
        Matrixf gb = g.colwise().sum();

        return vector<std::optional<Matrixf>>{gx, gw, gb};
    };

    return {ret, grad_func};
}

std::tuple<Matrixf, ManualDiffFunc> gauss(const Eigen::Ref<const Matrixf>& x)
{
    Matrixf ret = x.cwiseProduct(-x).array().exp();

    auto grad_func = [=](const Matrixf& grad) {
        // grad .* e^(-x^2) .* (-2x)
        Matrixf grad_x = grad.array().cwiseProduct(
            ret.array().cwiseProduct(-2.0 * x.array()));

        return vector<std::optional<Matrixf>>{grad_x};
    };

    return {ret, grad_func};
}

std::tuple<Matrixf, ManualDiffFunc> l2_loss(const Eigen::Ref<const Matrixf>& y_pred,
                                            const Eigen::Ref<const Matrixf>& y)
{
    float batchsize = y_pred.rows();
    Matrixf diff = y_pred - y;

    Matrixf loss(1, 1);
    float loss_val = (1.0f / batchsize) * diff.array().pow(2.0).sum();
    loss(0, 0) = loss_val;

    auto grad_func = [=](const Matrixf& grad) {
        Matrixf grad_loss = grad.replicate(diff.rows(), diff.cols())
                                .cwiseProduct((1.0f / batchsize) * (2.0 * diff));
        return vector<std::optional<Matrixf>>{grad_loss};
    };

    return {loss, grad_func};
}

Matrixf RBFNetwork::forward(const Eigen::Ref<const Matrixf>& x)
{
    Matrixf x1;
    ManualDiffFunc g1;
    std::tie(x1, g1) = fc(x, w1, b1);

    Matrixf h;
    ManualDiffFunc gg;
    std::tie(h, gg) = gauss(x1);

    Matrixf x2;
    ManualDiffFunc g2;
    std::tie(x2, g2) = fc(h, w2, b2);

    return x2;
}

Matrixf RBFNetwork::forward_backward(std::shared_ptr<Optimizer> opt,
                                     const Eigen::Ref<const Matrixf>& x,
                                     const Eigen::Ref<const Matrixf>& y)
{
    Matrixf x1;
    ManualDiffFunc g1;
    std::tie(x1, g1) = fc(x, w1, b1);

    Matrixf h;
    ManualDiffFunc gg;
    std::tie(h, gg) = gauss(x1);

    Matrixf x2;
    ManualDiffFunc g2;
    std::tie(x2, g2) = fc(h, w2, b2);

    Matrixf loss;
    ManualDiffFunc gloss;
    std::tie(loss, gloss) = l2_loss(x2, y);

    Matrixf gloss_in_grad = Matrixf::Ones(loss.rows(), loss.cols());

    auto gloss_ret = gloss(gloss_in_grad);
    Matrixf& g2_in_grad = gloss_ret[0].value();

    auto g2_ret = g2(g2_in_grad);
    Matrixf& gg_in_grad = g2_ret[0].value();

    auto gg_ret = gg(gg_in_grad);
    Matrixf& g1_in_grad = gg_ret[0].value();

    auto g1_ret = g1(g1_in_grad);

    vector<Matrixf> grads;
    grads.emplace_back(std::move(g1_ret[1].value())); // ∂ loss / ∂ w1
    grads.emplace_back(std::move(g1_ret[2].value())); // ∂ loss / ∂ b1
    grads.emplace_back(std::move(g2_ret[1].value())); // ∂ loss / ∂ w2
    grads.emplace_back(std::move(g2_ret[2].value())); // ∂ loss / ∂ b2

    opt->update_params({&w1, &b1, &w2, &b2}, grads);

    return loss;
}

void RBFNetwork::fit(std::shared_ptr<Optimizer> opt, const std::vector<Point>& points)
{
    norm = Normalizer(points);

    init(opt);

    Matrixf X(points.size(), 1);
    Matrixf Y(points.size(), 1);

    for (int i = 0; i < points.size(); ++i) {
        X(i, 0) = norm.normalize_x(points[i].x);
        Y(i, 0) = norm.normalize_y(points[i].y);
    }

    for (int i = 0; i < 1000; ++i) {
        Matrixf loss = forward_backward(opt, X, Y);
        std::cout << ">> loss: " << loss << std::endl;
    }
}

std::vector<Point> RBFNetwork::predict(float x_start, float x_end, int num_points)
{
    auto step = (x_end - x_start) / (num_points - 1);
    float x = x_start;

    std::vector<Point> ret;
    ret.resize(num_points);

    Matrixf X(num_points, 1);
    for (int i = 0; i < num_points; i++, x += step) {
        ret[i].x = x;
        X(i, 0) = norm.normalize_x(x);
    }
    Matrixf Y_pred = forward(X);
    for (int i = 0; i < num_points; i++) {
        ret[i].y = norm.denormalize_y(Y_pred(i, 0));
    }
    return ret;
}
