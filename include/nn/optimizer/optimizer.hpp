#pragma once
#include <nn/math/matrix.hpp>

namespace optimizer {
    class Optimizer {
    protected:
        Optimizer() = default;
    public:
        virtual ~Optimizer() = default;
        virtual void operator()(const std::vector<const math::Matrix &>&grads, const std::vector<math::Matrix *>& weights) = 0;
        virtual void reset() = 0;
    };
} // namespace optimizer