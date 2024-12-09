#pragma once
#include <nn/layer/layer.hpp>

namespace layer {
    class ReLu final : public Layer {
    public:
        ReLu(size_t batch_size, size_t input_size);

        const math::Matrix &forward(math::Matrix &&in) override;

        const math::Matrix &forward(const math::Matrix &in) override;

        const math::Matrix &backward(math::Matrix &&dout) override;

        const math::Matrix &backward(const math::Matrix &dout) override;

        math::Matrix *weights() override;
    };
} // namespace layer
