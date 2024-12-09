#pragma once
#include "optimizer.hpp"

namespace optimizer {
    class StochasticGradientDescent final : public Optimizer{
        float learning_rate = 0.001;
        float momentum = 0.9;
        std::vector<math::Matrix> velocities = std::vector<math::Matrix>();

    public:
        StochasticGradientDescent() = default;
        void operator()(const std::vector<const math::Matrix &>&grads, const std::vector<math::Matrix *>& weights) override;
        void reset() override {
            velocities = std::vector<math::Matrix>();
        }
    };
} // namespace optimizer
