#pragma once
#include "../../math/matrix.hpp"

namespace layer {
    class Layer { // abstract class representing a single network layer
    protected:
        size_t _batch_size;
        size_t _input_size;
        math::Matrix _cache;
        Layer(size_t batch_size, size_t input_size);
    public:
        virtual ~Layer() = default;
        virtual math::Matrix* forward(math::Matrix &&in) = 0;
        virtual math::Matrix backward(math::Matrix &&dout) = 0;
    };
} // namespace layer