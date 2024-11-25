#pragma once
#include <loss.hpp>

namespace loss {
    class CrossEntropy final : public Loss {
        math::Matrix cache;
    public:
        CrossEntropy(const std::vector<size_t> &labels, size_t num_classes);
        float forward(const math::Matrix &y_out) override;
        math::Matrix backward() const;
        math::Matrix backward(const math::Matrix &y_out) override;
    };
} // namespace loss