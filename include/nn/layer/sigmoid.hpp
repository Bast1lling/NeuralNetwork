#pragma once
#include <nn/layer/layer.hpp>

namespace layer {
    class Sigmoid final : public Layer {
    public:
        Sigmoid(size_t batch_size, size_t input_size);
        math::Matrix* forward(math::Matrix &&in) override;
        math::Matrix backward(math::Matrix &&dout) override;
    };
} // namespace layer
