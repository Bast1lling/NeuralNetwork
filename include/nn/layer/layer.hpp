#pragma once
#include <nn/math/matrix.hpp>

namespace layer {
    class Layer { // abstract class representing a single network layer
    protected:
        size_t _batch_size;
        size_t _input_size;
        math::Matrix _cache;
        math::Matrix _grad = math::Matrix();
        Layer(size_t batch_size, size_t input_size);
    public:
        virtual ~Layer() = default;
        virtual const math::Matrix &forward(math::Matrix &&in) = 0;
        virtual const math::Matrix &forward(const math::Matrix &in) = 0;
        virtual const math::Matrix &backward(math::Matrix &&dout) = 0;
        virtual const math::Matrix &backward(const math::Matrix &dout) = 0;
        virtual math::Matrix *weights() = 0;
    };
} // namespace layer