#pragma once
#include <loss.hpp>

namespace loss {
    class L2 final : public Loss {
    public:
        explicit L2(math::Matrix &&y_truth);
        float forward(const math::Matrix &y_out) override;
        math::Matrix backward(const math::Matrix &y_out) override;
    };
} // namespace loss