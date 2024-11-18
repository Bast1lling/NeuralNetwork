#include "network.hpp"

#include <matrix.hpp>
#include <vector>

namespace network {
    class Network {
    public:
        virtual ~Network() = default;

        virtual math::Matrix forward(math::Matrix input);
        virtual math::Matrix backward(math::Matrix output);
    };
} // namespace network
