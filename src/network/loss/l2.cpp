#include "l2.hpp"

namespace loss {
    L2::L2(math::Matrix &&y_truth) : Loss(y_truth) {
    }

    float L2::forward(const math::Matrix &y_out) {
        return (y_out - y_truth).pow(2.f).mean().mean();
    }

    math::Matrix L2::backward(const math::Matrix &y_out) {
        return (y_out - y_truth) * 2;
    }
} // namespace loss
