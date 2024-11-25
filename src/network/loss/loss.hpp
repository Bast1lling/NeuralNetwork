#pragma once
#include <matrix.hpp>

namespace loss {
    class Loss {
    protected:
        math::Matrix y_truth{};
        explicit Loss(math::Matrix y_truth);
        Loss();
    public:
        virtual ~Loss() = default;

        virtual float forward(const math::Matrix &y_out) = 0;
        virtual math::Matrix backward(const math::Matrix &y_out) = 0;
    };
} // namespace loss