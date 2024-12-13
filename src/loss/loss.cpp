#include <nn/loss/loss.hpp>

namespace loss {
    std::pair<float, math::Matrix> l1(const math::Matrix &y_out, const math::Matrix &y_truth) {
        math::Matrix diff = y_out - y_truth;
        float loss = diff.abs().mean().mean();
        auto new_data = std::vector<std::vector<float> >(diff.n());
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
        return std::make_pair(loss, math::Matrix(new_data));
    }

    std::pair<float, math::Matrix> l2(const math::Matrix &y_out, const math::Matrix &y_truth) {
        math::Matrix diff = y_out - y_truth;
        float loss = diff.pow(2.f).mean().mean();
        return std::make_pair(loss, diff * 2);
    }

    std::pair<float, math::Matrix> cross_entropy(const math::Matrix &y_out, const math::Matrix &y_truth) {
        auto soft_maxed_data = std::vector<math::Vector>(y_out.n());
        for (size_t i = 0; i < soft_maxed_data.size(); i++) {
            const math::Vector &row = y_out[i];
            float max = row.max();
            math::Vector temp = (row - max).exp();
            soft_maxed_data[i] = temp / temp.sum();
        }
        math::Matrix cache = math::Matrix(y_out.n(), y_out.m(), soft_maxed_data);
        float loss = (-y_truth.had(cache.log())).sum().mean();
        math::Matrix grad = (cache - y_truth) / static_cast<float>(y_truth.n());
        return std::make_pair(loss, grad);
    }

} // namespace loss
