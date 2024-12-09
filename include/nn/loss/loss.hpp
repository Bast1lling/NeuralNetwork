#pragma once
#include <memory>
#include <nn/math/matrix.hpp>

namespace loss {
    using LossFunction = std::pair<float, math::Matrix>(const math::Matrix&, const math::Matrix&);
    std::pair<float, math::Matrix> l1(const math::Matrix&, const math::Matrix&);
    std::pair<float, math::Matrix> l2(const math::Matrix&, const math::Matrix&);
    std::pair<float, math::Matrix> cross_entropy(const math::Matrix&, const math::Matrix&);
} // namespace loss