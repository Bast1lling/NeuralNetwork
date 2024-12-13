#include <nn/optimizer/sgd.hpp>
#include <cassert>
namespace optimizer {
    void StochasticGradientDescent::operator()(const std::vector<const math::Matrix *> &grads,
        const std::vector<math::Matrix *> &weights) {
        assert(grads.size() - 1 == weights.size() && "There is not exactly one gradient per layer");
        if (velocities.empty()) {
            for (size_t i = 0; i < weights.size(); i++) {
                if (weights[i]) { // if layer has weights to be trained
                    math::Matrix velocity = -*grads[i + 1] * learning_rate;
                    // update weight
                    *weights[i] = *weights[i] + velocity;
                    // save velocity for next update
                    velocities.emplace_back(velocity);
                }
            }
        }
        else {
            size_t counter = 0;
            for (size_t i = 0; i <weights.size(); i++) {
                if (weights[i]) {
                    velocities[counter] = velocities[counter] * momentum + -*grads[i + 1] * learning_rate;
                    *weights[i] = *weights[i] + velocities[counter];
                    ++counter;
                }
            }
        }
    }
} // namespace optimizer