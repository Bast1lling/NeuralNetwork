#pragma once
#include <layer.hpp>

namespace layer {
    class ReLu final : public Layer {
    public:
        ReLu(size_t batch_size, size_t input_size);
        math::Matrix* forward(math::Matrix &&in) override;
        math::Matrix backward(math::Matrix &&dout) override;
    };
} // namespace layer
