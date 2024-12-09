#include <nn/layer/layer.hpp>

namespace layer {
    Layer::Layer(size_t batch_size, size_t input_size) : _batch_size(batch_size), _input_size(input_size), _cache(math::Matrix(batch_size, input_size)) {

    }
} // namespace layer
