#pragma once

#include "gui.hpp"

#include "Eigen/Core"

#include <variant>
#include <vector>

using Matrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vectorf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using ParamT = std::variant<Matrixf, Vectorf>;
using ParamTP = std::variant<Matrixf*, Vectorf*>;

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

struct Optimizer
{
    float lr;

    Optimizer(float lr)
        : lr{lr} {};
    virtual ~Optimizer(){};
    virtual void init_state(const std::vector<ParamTP>& params) = 0;
    virtual void update_params(const std::vector<ParamTP>& params,
                               const std::vector<ParamT>& grads) = 0;
};

struct SgdOptimizer : public Optimizer
{
    SgdOptimizer(float lr);
    ~SgdOptimizer() override{};
    void init_state(const std::vector<ParamTP>& params) override;
    void update_params(const std::vector<ParamTP>& params, const std::vector<ParamT>& grads) override;
};

struct AdamOptimizer : public Optimizer
{
    float b1;
    float b2;
    float eps;
    std::vector<ParamT> m;
    std::vector<ParamT> v;

    AdamOptimizer(float lr, float b1=0.9f, float b2=0.999f, float eps=1e-8f);
    ~AdamOptimizer() override{};
    void init_state(const std::vector<ParamTP>& params) override;
    void update_params(const std::vector<ParamTP>& params, const std::vector<ParamT>& grads) override;
};

struct RBFNetwork
{
    Normalizer norm;

    int num_basis;
    Matrixf w1;
    Matrixf b1;
    Matrixf w2;
    Matrixf b2;

    RBFNetwork(int num_basis = 0);
    void init(std::shared_ptr<Optimizer> opt);
    void fit(std::shared_ptr<Optimizer> opt, const std::vector<Point>& points);
    std::vector<Point> predict(float x_start, float x_end, int num_points);

private:
    Matrixf forward(const Eigen::Ref<const Matrixf>& x);
    Matrixf forward_backward(std::shared_ptr<Optimizer> opt, const Eigen::Ref<const Matrixf>& x,
                             const Eigen::Ref<const Matrixf>& y);
};
