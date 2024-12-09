#pragma once
#include <functional>
#include <memory>
#include <vector>
#include <nn/layer/layer.hpp>
#include <nn/loss/loss.hpp>
#include <nn/optimizer/optimizer.hpp>

namespace network {
    class Network {
        std::vector<math::Matrix *> weights;
        // process model input and return the generated output
        const math::Matrix &forward(const math::Matrix &in);
        // compute layer-wise gradients according to loss function gradient
        std::vector<const math::Matrix &> backward(math::Matrix &&dloss);

    public:
        std::vector<std::unique_ptr<layer::Layer> > model;
        loss::LossFunction loss_function = loss::cross_entropy;
        optimizer::Optimizer optimizer = optimizer::sgd_step;

        Network(size_t batch_size, size_t input_size, size_t output_size)();

        // train a single batch and return loss and accuracy
        float train(const math::Matrix &X, const math::Matrix &y);
    };
} // namespace network
