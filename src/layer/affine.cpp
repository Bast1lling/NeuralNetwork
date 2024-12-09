#include <nn/layer/affine.hpp>
#include <print>

namespace layer {
    Affine::Affine(const size_t batch_size, const size_t input_size,
                   const size_t output_size) : Layer(batch_size, input_size), _output_size(output_size),
                                               _in_cache(math::Matrix(batch_size, input_size)),
                                               _weights(math::Matrix::getRandom(input_size, output_size, 0.001)),
                                               _b(math::Vector(output_size)),
                                               _d_weights(math::Matrix(input_size, output_size)),
                                               _d_b(math::Vector(output_size)) {
    }

    Affine::Affine(math::Matrix &&weights, math::Vector &&b) : Layer(weights.m(), weights.n()), _output_size(b.size()),
                                                                _in_cache(weights.m(), weights.n()),
                                                               _weights(weights), _b(b),
                                                               _d_weights(weights.n(), weights.m()), _d_b(weights.m()) {
    }

    size_t Affine::output_size() const {
        return _output_size;
    }

    const math::Matrix &Affine::forward(math::Matrix &&in) {
        _in_cache = in;
        _cache = _in_cache * _weights + _b;
        return _cache;
    }

    const math::Matrix &Affine::forward(const math::Matrix &in) {
        _in_cache = in;
        _cache = _in_cache * _weights + _b;
        return _cache;
    }

    const math::Matrix &Affine::backward(math::Matrix &&dout) {
        _d_weights = _in_cache.T() * dout;
        // d_b => sum_up all rows of dout individually
        std::vector<float> d_b_data = std::vector<float>(_b.size());
        for (size_t i = 0; i < _b.size(); i++) {
            d_b_data[i] = dout.get_col(i).sum();
        }
        _d_b = math::Vector(d_b_data);
        _grad = dout * _weights.T();
        return _grad;
    }

    const math::Matrix &Affine::backward(const math::Matrix &dout) {
        _d_weights = _in_cache.T() * dout;
        // d_b => sum_up all rows of dout individually
        std::vector<float> d_b_data = std::vector<float>(_b.size());
        for (size_t i = 0; i < _b.size(); i++) {
            d_b_data[i] = dout.get_col(i).sum();
        }
        _d_b = math::Vector(d_b_data);
        _grad = dout * _weights.T();
        return _grad;
    }

    math::Matrix * Affine::weights() {
        // TODO: add bias into weights
        return &_weights;
    }

    void Affine::print() const {
        std::println("Weight-matrix derivative:");
        _d_weights.print();
        std::println("B-vector derivative:");
        _d_b.print();
    }
} // namespace layer
