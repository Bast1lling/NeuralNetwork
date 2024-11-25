#include "cross_entropy_loss.hpp"

namespace loss {
    CrossEntropy::CrossEntropy(const std::vector<size_t> &labels, size_t num_classes) : Loss(math::Matrix(1, 1)), cache(math::Matrix(labels.size(), num_classes)){
        auto data = std::vector<std::vector<float>>(labels.size());
        for (size_t i = 0; i < labels.size(); i++) {
            auto row = std::vector<float>(num_classes);
            row[labels[i]] = 1.0f;
            data[i] = row;
        }
        y_truth = math::Matrix(data);
    }

    float CrossEntropy::forward(const math::Matrix &y_out) {
        auto soft_maxed_data = std::vector<math::Vector>(y_out.n());
        for (size_t i = 0; i < y_out.n(); i++) {
            const math::Vector& row = y_out[i];
            float max = row.max();
            float mean = row.mean();
            soft_maxed_data[i] = ((row - max) / mean);
        }
        cache = math::Matrix(y_out.n(), y_out.m(), soft_maxed_data);
        return (-y_truth.had(cache.log())).sum().mean();
    }

    math::Matrix CrossEntropy::backward() const {
        return (cache - y_truth) / static_cast<float>(y_truth.n());
    }

    math::Matrix CrossEntropy::backward(const math::Matrix &y_out) {
        auto soft_maxed_data = std::vector<math::Vector>(y_out.n());
        for (size_t i = 0; i < y_out.n(); i++) {
            const math::Vector& row = y_out[i];
            float max = row.max();
            float mean = row.mean();
            soft_maxed_data[i] = ((row - max) / mean);
        }
        cache = math::Matrix(y_out.n(), y_out.m(), soft_maxed_data);
        return (cache - y_truth) / static_cast<float>(y_truth.n());
    }
} // namespace loss
