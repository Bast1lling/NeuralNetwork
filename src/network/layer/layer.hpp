#pragma once
#include "../../math/matrix.hpp"

namespace layer {
    class Layer { // abstract class representing a single network layer
    protected:
        size_t _n{};
        explicit Layer(size_t n);
    public:
        virtual ~Layer() = default;
        virtual math::Matrix* forward(math::Matrix &&in) = 0;
        virtual math::Matrix backward(math::Matrix &&dout) = 0;
    };
} // namespace layer