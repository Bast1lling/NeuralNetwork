#include "loss.hpp"

#include <utility>

namespace loss {
    Loss::Loss(math::Matrix y_truth) : y_truth(std::move(y_truth)) {};

    Loss::Loss() = default;
} // namespace loss