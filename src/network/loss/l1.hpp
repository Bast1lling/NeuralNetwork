#pragma once
#include <loss.hpp>

namespace loss {
    class L1 final : public Loss {
    public:
        explicit L1(math::Matrix &&y_truth);
        float forward(const math::Matrix &y_out) override;
        math::Matrix backward(const math::Matrix &y_out) override;
    };
} // namespace loss
