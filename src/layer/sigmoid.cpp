#include <nn/layer/sigmoid.hpp>

namespace layer {
    Sigmoid::Sigmoid(const size_t batch_size, const size_t input_size) : Layer(batch_size, input_size) {
    }

    const math::Matrix &Sigmoid::forward(math::Matrix &&in) {
        _cache = ((-in).exp() + 1).inv();
        return _cache;
    }

    const math::Matrix &Sigmoid::forward(const math::Matrix &in) {
        _cache = ((-in).exp() + 1).inv();
        return _cache;
    }

    const math::Matrix &Sigmoid::backward(math::Matrix &&dout) {
        _grad = dout.had(_cache.had(-_cache + 1));
        return _grad;
    }

    const math::Matrix &Sigmoid::backward(const math::Matrix &dout) {
        _grad = dout.had(_cache.had(-_cache + 1));
        return _grad;
    }

    math::Matrix * Sigmoid::weights() {
        return nullptr;
    }
} // namespace layer
