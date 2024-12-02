#include <nn/layer/relu.hpp>

namespace layer {
    ReLu::ReLu(const size_t batch_size, const size_t input_size) : Layer(batch_size, input_size) {
    }

    math::Matrix * ReLu::forward(math::Matrix &&in) {
        _cache = in.max(0);
        return &_cache;
    }

    math::Matrix ReLu::backward(math::Matrix &&dout) {
        std::vector<std::vector<float>> data = std::vector<std::vector<float>>(dout.n());
        for (size_t i = 0; i < dout.n(); i++) {
            std::vector<float> row = std::vector<float>(dout.m());
            for (size_t j = 0; j < dout.m(); j++) {
                const float f = dout[i][j];
                if (float g = _cache[i][j]; g <= 0) {
                    row[j] = 0;
                }
                else {
                    row[j] = f;
                }
            }
            data[i] = row;
        }
        return math::Matrix(data);
    }
} // namespace layer
