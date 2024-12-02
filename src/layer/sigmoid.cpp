#include <nn/layer/sigmoid.hpp>

namespace layer {
    Sigmoid::Sigmoid(const size_t batch_size, const size_t input_size) : Layer(batch_size, input_size) {
    }

    math::Matrix* Sigmoid::forward(math::Matrix &&in) {
        _cache = ((-in).exp() + 1).inv();
        return &_cache;
    }

    math::Matrix Sigmoid::backward(math::Matrix &&dout) {
        return dout.had(_cache.had(-_cache + 1));
    }
} // namespace layer
