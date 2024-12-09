#pragma once
#include <nn/math/matrix.hpp>

namespace optimizer {
    class OptimizerParameters {

    };
    using Optimizer = void(std::vector<const math::Matrix &>, std::vector<math::Matrix *>, OptimizerParameters);
    void sgd_step(std::vector<const math::Matrix &>grads, std::vector<math::Matrix *> weights, OptimizerParameters params);
} // namespace optimizer