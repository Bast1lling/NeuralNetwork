#pragma once
#include <nn/layer/layer.hpp>

namespace layer {
    class Affine final : public Layer {
        size_t _output_size;
        math::Matrix _in_cache;
        math::Matrix _weights;
        math::Vector _b;
        math::Matrix _d_weights;
        math::Vector _d_b;
    public:
        // Constructors
        // n = batch-size | d = input-neurons | m = output-neurons
        Affine(size_t batch_size, size_t input_size, size_t output_size);
        Affine(math::Matrix &&weights, math::Vector &&b);
        // Getters
        size_t output_size() const;

        // Virtual
        const math::Matrix &forward(math::Matrix &&in) override;

        const math::Matrix &forward(const math::Matrix &in) override;

        const math::Matrix &backward(math::Matrix &&dout) override;

        const math::Matrix &backward(const math::Matrix &dout) override;

        math::Matrix *weights() override;

        // Others
        void print() const;
    };
} // namespace layer
