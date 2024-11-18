#include "sigmoid.hpp"

namespace layer {
    Sigmoid::Sigmoid(const size_t n, const size_t m) : Layer(n), cache(math::Matrix(n, m)){
    }

    math::Matrix* Sigmoid::forward(math::Matrix &&in) {
        cache = ((-in).exp() + 1).inv();
        return &cache;
    }

    math::Matrix Sigmoid::backward(math::Matrix &&dout) {
        return dout.had(cache.had(-cache + 1));
    }
} // namespace layer
