#include "l1.hpp"

namespace loss {
    L1::L1(math::Matrix &&y_truth) : Loss(y_truth){

    }

    float L1::forward(const math::Matrix &y_out) {
        return (y_out - y_truth).abs().mean().mean();
    }

    math::Matrix L1::backward(const math::Matrix &y_out) {
        math::Matrix diff = y_out - y_truth;
        std::vector<std::vector<float>> new_data = std::vector<std::vector<float>>(diff.n());
        for (size_t i = 0; i < diff.n(); i++) {
            std::vector<float> row = std::vector<float>(diff.m());
            for (size_t j = 0; j < diff.m(); j++) {
                float v = diff[i][j];
                if (v == 0) {
                    row[j] = 0;
                    continue;
                }
                row[j] = (v < 0) ? -1 : 1;
            }
            new_data[i] = row;
        }
        return math::Matrix(new_data);
    }
} // namespace loss