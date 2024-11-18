#pragma once
#include <layer.hpp>

namespace layer {
    class Sigmoid final : public Layer {
        math::Matrix cache;
    public:
        explicit Sigmoid(size_t n, size_t m);
        math::Matrix* forward(math::Matrix &&in) override;
        math::Matrix backward(math::Matrix &&dout) override;
    };
} // namespace layer
